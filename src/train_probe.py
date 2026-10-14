"""Baseline classifier on FROZEN embeddings: the number fine-tuning has to beat.

Trains a linear or small-MLP head on embeddings_<tag>.npy (train split), picks
the epoch with the best val macro-F1, and reports on the held-out viz split.
Splits are per recording, so test clips come from recordings never seen in
training.

Metrics, all on viz:
  clip accuracy      -- top-1 per 5 s clip
  macro-F1           -- every class weighs the same, so the 98% bird majority
                        cannot hide failures on frogs, insects and rare birds
  recording accuracy -- logits averaged over a recording's clips, then top-1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config


def evaluate(logits: np.ndarray, y: np.ndarray, rec: np.ndarray, n_cls: int) -> dict:
    pred = logits.argmax(1)
    df = pd.DataFrame(logits)
    df["rec"] = rec
    rec_logits = df.groupby("rec").mean()
    rec_y = pd.Series(y, index=rec).groupby(level=0).first().loc[rec_logits.index]
    return {
        "clip_acc": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=np.unique(y), zero_division=0)),
        "rec_acc": float((rec_logits.to_numpy().argmax(1) == rec_y.to_numpy()).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Linear/MLP probe on frozen embeddings.")
    ap.add_argument("--tag", default="bc26")
    ap.add_argument("--head", choices=("linear", "mlp"), default="linear")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--balanced", action="store_true",
                    help="weight the loss by inverse class frequency")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    X = np.load(config.EMB_DIR / f"embeddings_{args.tag}.npy").astype(np.float32)
    meta = pd.read_csv(config.EMB_DIR / f"clip_order_{args.tag}.csv",
                       dtype={"species_code": str}).fillna("")
    classes = sorted(meta["species_code"].unique())
    idx = {c: i for i, c in enumerate(classes)}
    y = meta["species_code"].map(idx).to_numpy()
    split = meta["split"].to_numpy()
    rec = meta["recording_id"].to_numpy()

    # standardise with train statistics only
    tr = split == "train"
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    X = (X - mu) / sd

    # a test class never seen in training cannot be predicted; report it, keep it
    unseen = sorted(set(y[split == "viz"]) - set(y[tr]))
    print(f"classes: {len(classes)}  |  viz classes absent from train: {len(unseen)}")

    dev = args.device
    Xt = {s: torch.tensor(X[split == s], device=dev) for s in ("train", "val", "viz")}
    yt = torch.tensor(y[tr], device=dev)
    d, k = X.shape[1], len(classes)
    head = (nn.Linear(d, k) if args.head == "linear" else
            nn.Sequential(nn.Linear(d, 1024), nn.GELU(), nn.Dropout(0.3), nn.Linear(1024, k))).to(dev)

    weight = None
    if args.balanced:
        counts = np.bincount(y[tr], minlength=k).astype(np.float32)
        weight = torch.tensor(np.where(counts > 0, counts.sum() / (k * np.maximum(counts, 1)), 0),
                              device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    def logits_for(s: str) -> np.ndarray:
        head.eval()
        with torch.no_grad():
            return head(Xt[s]).float().cpu().numpy()

    best, best_state, n = -1.0, None, len(yt)
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 1024):
            b = perm[i:i + 1024]
            opt.zero_grad()
            loss_fn(head(Xt["train"][b]), yt[b]).backward()
            opt.step()
        sched.step()
        m = evaluate(logits_for("val"), y[split == "val"], rec[split == "val"], k)
        if m["macro_f1"] > best:
            best, best_state = m["macro_f1"], {kk: v.clone() for kk, v in head.state_dict().items()}
            print(f"  epoch {epoch + 1:>3}  val clip_acc {m['clip_acc']:.3f}  "
                  f"macro_f1 {m['macro_f1']:.3f}  rec_acc {m['rec_acc']:.3f}", flush=True)

    head.load_state_dict(best_state)
    result = {"tag": args.tag, "head": args.head, "balanced": args.balanced,
              "val": evaluate(logits_for("val"), y[split == "val"], rec[split == "val"], k),
              "test_viz": evaluate(logits_for("viz"), y[split == "viz"], rec[split == "viz"], k)}
    print(json.dumps(result, indent=2))

    out = config.EXPORT_DIR / args.tag / f"probe_{args.head}{'_bal' if args.balanced else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
