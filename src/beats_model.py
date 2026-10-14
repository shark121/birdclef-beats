"""Phase 3a: load a pretrained BEATs checkpoint as a fixed feature extractor.

The BEATs *code* lives in microsoft/unilm (MIT). The official *checkpoint*
links in that repo point at a decommissioned MSR blob host, so we pull the
weights from a HuggingFace mirror of the same files. Both are fetched once and
cached under data/checkpoints/.

We default to BEATs_iter3_plus_AS2M.pt -- the self-supervised iter3 model, not
one fine-tuned on AudioSet labels. For a species-clustering question we want
general audio representations, not a 527-class AudioSet decision boundary
baked into the features.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.sources import download

UPSTREAM = "https://raw.githubusercontent.com/microsoft/unilm/master/beats"
CODE_FILES = ("BEATs.py", "backbone.py", "modules.py", "quantizer.py", "Tokenizers.py")

CHECKPOINTS = {
    # name -> (huggingface repo, filename)
    "iter3_plus_AS2M": ("lpepino/beats_ckpts", "BEATs_iter3_plus_AS2M.pt"),
    "iter3": ("lpepino/beats_ckpts", "BEATs_iter3.pt"),
    "tokenizer_iter3_plus_AS2M": ("mooneyko/BEATs", "Tokenizer_iter3_plus_AS2M.pt"),
}
DEFAULT_CHECKPOINT = "iter3_plus_AS2M"


def fetch_code(dest: Path | None = None) -> Path:
    """Download the upstream BEATs modules next to each other and return the dir."""
    dest = dest or (config.CKPT_DIR / "beats_upstream")
    dest.mkdir(parents=True, exist_ok=True)
    for name in CODE_FILES:
        target = dest / name
        if target.exists() and target.stat().st_size > 1000:
            continue
        if not download(f"{UPSTREAM}/{name}", target, min_bytes=500):
            raise RuntimeError(f"could not fetch BEATs source file: {name}")
    return dest


def fetch_checkpoint(name: str = DEFAULT_CHECKPOINT) -> Path:
    repo, filename = CHECKPOINTS[name]
    target = config.CKPT_DIR / filename
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    if not download(url, target, timeout=600, min_bytes=10_000_000):
        raise RuntimeError(f"could not fetch checkpoint {filename} from {repo}")
    return target


class BeatsEmbedder:
    """Wraps BEATs so callers only ever see: waveforms in, one vector out."""

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device: str = "cpu",
                 pooling: str = "mean"):
        code_dir = fetch_code()
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        from BEATs import BEATs, BEATsConfig            # noqa: E402  (needs sys.path)

        ckpt_path = fetch_checkpoint(checkpoint)
        # weights_only defaults to True on torch>=2.6; this checkpoint carries a
        # cfg dict alongside the tensors, so it must be loaded in full.
        blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg = BEATsConfig(blob["cfg"])
        model = BEATs(cfg)
        model.load_state_dict(blob["model"])
        model.eval().to(device)

        self.model = model
        self.device = device
        self.pooling = pooling
        self.checkpoint_name = checkpoint
        self.embed_dim = int(cfg.encoder_embed_dim)

    @torch.inference_mode()
    def embed(self, waveforms: torch.Tensor) -> torch.Tensor:
        """(B, samples) float32 @16 kHz -> (B, embed_dim) float32."""
        waveforms = waveforms.to(self.device, dtype=torch.float32)
        padding_mask = torch.zeros(waveforms.shape, dtype=torch.bool,
                                   device=self.device)
        feats = self.model.extract_features(waveforms, padding_mask=padding_mask)[0]
        if feats.dim() == 2:                       # already pooled
            return feats.float().cpu()
        if self.pooling == "max":
            pooled = feats.max(dim=1).values
        else:
            pooled = feats.mean(dim=1)             # plan's default: mean-pool over time
        return pooled.float().cpu()


if __name__ == "__main__":
    import numpy as np
    emb = BeatsEmbedder()
    dummy = torch.from_numpy(
        np.random.randn(2, int(config.CLIP_SECONDS * config.SAMPLE_RATE_HZ)).astype("float32")
    ) * 0.05
    out = emb.embed(dummy)
    print(f"checkpoint : {emb.checkpoint_name}")
    print(f"embed_dim  : {emb.embed_dim}")
    print(f"output     : {tuple(out.shape)}  finite={bool(torch.isfinite(out).all())}")
