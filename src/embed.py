"""Phase 3a/3b: batch-embed every clip with BEATs.

The interface is deliberately identical for both tracks: the bird-pretrained
model of Phase 3b is swapped in via --checkpoint and writes to a different
--tag. The export step (Phase 4) does not change at all -- only the source of
the vectors does, exactly as the plan intends.

Work is shard-checkpointed so an interrupted overnight run resumes instead of
starting over.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.beats_model import DEFAULT_CHECKPOINT, BeatsEmbedder

SHARD = 512


def load_batch(paths: list[Path], n_samples: int) -> torch.Tensor:
    """Read clips into one (B, samples) tensor. Clips are already uniform
    length, but we defensively pad/trim so a stray file cannot break a batch."""
    buf = np.zeros((len(paths), n_samples), dtype=np.float32)
    for i, path in enumerate(paths):
        audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        take = min(len(audio), n_samples)
        buf[i, :take] = audio[:take]
    return torch.from_numpy(buf)


def main() -> int:
    ap = argparse.ArgumentParser(description="Embed clips with BEATs.")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--tag", default="baseline",
                    help="names the output set, e.g. baseline (3a) or birdpretrained (3b)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0, help="0 = torch default")
    ap.add_argument("--splits", default="", help="comma list, e.g. viz  (default: all)")
    ap.add_argument("--corpus", default=None,
                    help="read data/interim/clips_<corpus>.csv instead of clips.csv, "
                         "to embed one of the band-floor variants")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="cuda or cpu (default: cuda when available)")
    ap.add_argument("--pooling", default="mean", choices=("mean", "max"))
    args = ap.parse_args()

    config.ensure_dirs()
    name = f"clips_{args.corpus}.csv" if args.corpus else "clips.csv"
    clips_path = config.INTERIM_DIR / name
    if not clips_path.exists():
        print(f"missing {clips_path} -- run preprocess.py first")
        return 1

    if args.threads:
        torch.set_num_threads(args.threads)

    clips = pd.read_csv(clips_path).fillna("")
    if args.splits:
        keep = {s.strip() for s in args.splits.split(",") if s.strip()}
        clips = clips[clips["split"].isin(keep)]
    # THE canonical order. Everything downstream aligns to this and nothing
    # re-sorts or re-filters after this point (plan section 3.5).
    clips = clips.sort_values("clip_id").reset_index(drop=True)
    print(f"embedding {len(clips)} clips  tag={args.tag}  checkpoint={args.checkpoint}")

    embedder = BeatsEmbedder(checkpoint=args.checkpoint, pooling=args.pooling,
                             device=args.device)
    print(f"loaded BEATs: embed_dim={embedder.embed_dim}  pooling={args.pooling} "
          f"device={args.device}")

    n_samples = int(config.CLIP_SECONDS * config.SAMPLE_RATE_HZ)
    shard_dir = config.EMB_DIR / "_shards" / args.tag
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Shards are resumable, which is only safe while the clip set is unchanged.
    # Fingerprint the exact ordered clip list; if it moved (more audio fetched,
    # different filter), drop the old shards rather than silently splicing
    # vectors from one corpus onto the metadata of another.
    fingerprint = hashlib.sha1(
        "\n".join(clips["clip_id"].astype(str)).encode()).hexdigest()
    fp_file = shard_dir / "clipset.sha1"
    stale = (not fp_file.exists()) or fp_file.read_text().strip() != fingerprint
    if stale and any(shard_dir.glob("*.npy")):
        # A missing fingerprint means the shards predate this bookkeeping, so
        # their provenance is unknown -- which is exactly how run 1's shards got
        # spliced onto run 2's clip list. Unknown provenance counts as stale.
        print("  shard set is stale or of unknown provenance -- discarding")
        shutil.rmtree(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)
    fp_file.write_text(fingerprint)

    started = time.time()
    n_shards = (len(clips) + SHARD - 1) // SHARD
    for shard_idx in range(n_shards):
        shard_path = shard_dir / f"{shard_idx:05d}.npy"
        if shard_path.exists():
            continue                                   # resume: already done
        chunk = clips.iloc[shard_idx * SHARD:(shard_idx + 1) * SHARD]
        vectors = []
        for start in range(0, len(chunk), args.batch_size):
            batch = chunk.iloc[start:start + args.batch_size]
            paths = [config.CLIPS_DIR / p for p in batch["clip_path"]]
            vectors.append(embedder.embed(load_batch(paths, n_samples)).numpy())
        np.save(shard_path, np.concatenate(vectors, axis=0).astype(np.float32))
        done = min((shard_idx + 1) * SHARD, len(clips))
        rate = done / max(time.time() - started, 1e-6)
        print(f"  shard {shard_idx + 1}/{n_shards}  {done}/{len(clips)} clips  "
              f"{rate:.1f} clips/s  eta {(len(clips) - done) / max(rate, 1e-6) / 60:.1f} min",
              flush=True)

    embeddings = np.concatenate(
        [np.load(shard_dir / f"{i:05d}.npy") for i in range(n_shards)], axis=0)

    assert len(embeddings) == len(clips), \
        f"embedding/clip count mismatch: {len(embeddings)} vs {len(clips)}"
    if not np.isfinite(embeddings).all():
        bad = int((~np.isfinite(embeddings)).any(axis=1).sum())
        print(f"WARNING: {bad} embeddings contain NaN/Inf and will fail export validation")

    np.save(config.EMB_DIR / f"embeddings_{args.tag}.npy", embeddings)
    clips.to_csv(config.EMB_DIR / f"clip_order_{args.tag}.csv", index=False)

    print(f"\nembeddings: {embeddings.shape}  dtype={embeddings.dtype}")
    print(f"finite    : {bool(np.isfinite(embeddings).all())}")
    print(f"elapsed   : {(time.time() - started) / 60:.1f} min")
    print(f"wrote     : {config.EMB_DIR / f'embeddings_{args.tag}.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
