"""Fine-tune BEATs end to end for species classification.

Same clips, same per-recording splits and same metrics as train_probe.py, so
the two numbers are directly comparable: train on `train`, pick the best epoch
by val macro-F1, report once on the held-out `viz` split.

Choices, each aimed at accuracy on a heavily imbalanced 206-class set:
  * layer-wise LR decay -- deep layers move less than the top of the network,
    which keeps the pretrained features while the head and top layers adapt
  * sqrt-balanced sampling -- birds are 98% of clips; full inverse-frequency
    weighting over-fits the 1-clip classes (it hurt the frozen probe), so
    sample by count^-0.5 as a middle ground
  * waveform augmentation on the GPU: random circular shift, random gain, mixup
  * bf16 autocast; the BEATs fbank front end runs in fp32

At the end the pooled 768-d features of every clip are written to
embeddings_<out_tag>.npy, so the fine-tuned space can be analysed and
exported to the Projector exactly like the frozen one.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.beats_model import BeatsEmbedder
from src.train_probe import evaluate

N_SAMPLES = int(config.CLIP_SECONDS * config.SAMPLE_RATE_HZ)


class Clips(Dataset):
    def __init__(self, paths: list[str], labels: np.ndarray):
        self.paths, self.labels = paths, labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        audio, _ = sf.read(str(config.CLIPS_DIR / self.paths[i]), dtype="float32")
        buf = np.zeros(N_SAMPLES, dtype=np.float32)
        buf[:min(len(audio), N_SAMPLES)] = audio[:N_SAMPLES]
        return torch.from_numpy(buf), int(self.labels[i])


class Classifier(nn.Module):
    def __init__(self, beats: nn.Module, n_classes: int, dim: int = 768):
        super().__init__()
        self.beats = beats
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.2), nn.Linear(dim, n_classes))

    def features(self, wav: torch.Tensor) -> torch.Tensor:
        """BEATs.extract_features without the padding-mask path (clips are
        always full length), so it can run with gradients under autocast."""
        with torch.autocast("cuda", enabled=False):          # kaldi fbank wants fp32
            fbank = self.beats.preprocess(wav.float())
        x = self.beats.patch_embedding(fbank.unsqueeze(1))
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
        x = self.beats.layer_norm(x)
        if self.beats.post_extract_proj is not None:
            x = self.beats.post_extract_proj(x)
        x = self.beats.dropout_input(x)
        x, _ = self.beats.encoder(x, padding_mask=None)
        return x.mean(dim=1)                                  # same pooling as embed.py

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(wav)
        return self.head(feats), feats


def param_groups(model: Classifier, base_lr: float, head_lr: float, decay: float,
                 weight_decay: float) -> list[dict]:
    n_layers = len(model.beats.encoder.layers)
    groups: dict[tuple[float, bool], list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("head."):
            lr = head_lr
        else:
            m = re.search(r"encoder\.layers\.(\d+)\.", name)
            depth = int(m.group(1)) + 1 if m else (n_layers if "encoder.layer_norm" in name else 0)
            lr = base_lr * decay ** (n_layers - depth)
        no_wd = p.ndim == 1 or name.endswith(".bias")
        groups.setdefault((lr, no_wd), []).append(p)
    return [{"params": ps, "lr": lr, "initial_lr": lr, "weight_decay": 0.0 if no_wd else weight_decay}
            for (lr, no_wd), ps in groups.items()]


@torch.no_grad()
def predict(model: Classifier, loader: DataLoader, dev: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, feats = [], []
    for wav, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo, fe = model(wav.to(dev, non_blocking=True))
        logits.append(lo.float().cpu())
        feats.append(fe.float().cpu())
    return torch.cat(logits).numpy(), torch.cat(feats).numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune BEATs for species classification.")
    ap.add_argument("--tag", default="bc26", help="clip set (clip_order_<tag>.csv)")
    ap.add_argument("--out-tag", default="bc26_ft")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5, help="LR of the TOP encoder layer")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--layer-decay", type=float, default=0.75)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--mixup", type=float, default=0.5, help="probability of mixup per batch")
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after N steps")
    args = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(config.SPLIT_SEED)
    meta = pd.read_csv(config.EMB_DIR / f"clip_order_{args.tag}.csv",
                       dtype={"species_code": str}).fillna("")
    classes = sorted(meta["species_code"].unique())
    cidx = {c: i for i, c in enumerate(classes)}
    y = meta["species_code"].map(cidx).to_numpy()
    split = meta["split"].to_numpy()
    rec = meta["recording_id"].to_numpy()
    paths = meta["clip_path"].tolist()

    def loader(s: str, train: bool = False) -> DataLoader:
        ix = np.flatnonzero(split == s)
        ds = Clips([paths[i] for i in ix], y[ix])
        sampler = None
        if train:
            counts = np.bincount(y[ix], minlength=len(classes)).astype(np.float64)
            w = counts[y[ix]] ** -0.5
            sampler = WeightedRandomSampler(torch.tensor(w), num_samples=len(ix), replacement=True)
        return DataLoader(ds, batch_size=args.batch_size, sampler=sampler, shuffle=False,
                          num_workers=args.workers, pin_memory=True, drop_last=train,
                          persistent_workers=True)

    train_dl, val_dl, viz_dl = loader("train", True), loader("val"), loader("viz")

    beats = BeatsEmbedder(device=dev).model
    model = Classifier(beats, len(classes)).to(dev)
    opt = torch.optim.AdamW(param_groups(model, args.lr, args.head_lr, args.layer_decay,
                                         args.weight_decay), betas=(0.9, 0.98))
    total = args.epochs * len(train_dl)

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        t = (step - args.warmup_steps) / max(1, total - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    ckpt_dir = config.CKPT_DIR / args.out_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_f1, step, history = -1.0, 0, []
    print(f"classes {len(classes)}  train {len(train_dl.dataset)}  val {len(val_dl.dataset)}  "
          f"viz {len(viz_dl.dataset)}  steps/epoch {len(train_dl)}", flush=True)

    for epoch in range(args.epochs):
        model.train()
        t0, seen, run_loss = time.time(), 0, 0.0
        for wav, lab in train_dl:
            wav, lab = wav.to(dev, non_blocking=True), lab.to(dev, non_blocking=True)
            b = wav.shape[0]
            # --- augmentation, on the GPU ---------------------------------
            shifts = torch.randint(0, N_SAMPLES, (b,), device=dev)
            idx = (torch.arange(N_SAMPLES, device=dev)[None, :] + shifts[:, None]) % N_SAMPLES
            wav = torch.gather(wav, 1, idx)
            wav = wav * torch.empty(b, 1, device=dev).uniform_(0.3, 1.2)
            target = nn.functional.one_hot(lab, len(classes)).float()
            if torch.rand(1).item() < args.mixup:
                lam = float(np.random.beta(0.4, 0.4))
                perm = torch.randperm(b, device=dev)
                wav = lam * wav + (1 - lam) * wav[perm]
                target = lam * target + (1 - lam) * target[perm]
            target = target * 0.9 + 0.1 / len(classes)           # label smoothing
            # ----------------------------------------------------------------
            scale = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * scale
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(wav)
                loss = torch.sum(-target * torch.log_softmax(logits.float(), dim=1), dim=1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            seen += b
            run_loss += loss.item() * b
            if step % 100 == 0:
                print(f"  ep {epoch + 1} step {step}/{total}  loss {run_loss / seen:.3f}  "
                      f"{seen / (time.time() - t0):.0f} clips/s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break

        val_logits, _ = predict(model, val_dl, dev)
        m = evaluate(val_logits, y[split == "val"], rec[split == "val"], len(classes))
        history.append({"epoch": epoch + 1, "train_loss": run_loss / max(seen, 1), **m})
        print(f"epoch {epoch + 1}: val clip_acc {m['clip_acc']:.3f}  macro_f1 {m['macro_f1']:.3f}  "
              f"rec_acc {m['rec_acc']:.3f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            torch.save({"model": model.state_dict(), "classes": classes, "args": vars(args),
                        "epoch": epoch + 1, "val": m}, ckpt_dir / "best.pt")
        if args.max_steps and step >= args.max_steps:
            break

    # ---- final: best checkpoint, report on viz, dump fine-tuned embeddings ---
    state = torch.load(ckpt_dir / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(state["model"])
    viz_logits, _ = predict(model, viz_dl, dev)
    result = {"out_tag": args.out_tag, "best_epoch": state["epoch"], "val": state["val"],
              "test_viz": evaluate(viz_logits, y[split == "viz"], rec[split == "viz"], len(classes)),
              "history": history, "args": vars(args)}
    print(json.dumps({k: result[k] for k in ("best_epoch", "val", "test_viz")}, indent=2))

    out_dir = config.EXPORT_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "finetune_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    all_dl = DataLoader(Clips(paths, y), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    _, feats = predict(model, all_dl, dev)
    np.save(config.EMB_DIR / f"embeddings_{args.out_tag}.npy", feats.astype(np.float32))
    shutil.copy(config.EMB_DIR / f"clip_order_{args.tag}.csv",
                config.EMB_DIR / f"clip_order_{args.out_tag}.csv")
    print(f"wrote embeddings_{args.out_tag}.npy {feats.shape} and {out_dir / 'finetune_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
