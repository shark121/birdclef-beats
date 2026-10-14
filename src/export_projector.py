"""Phase 4: write the Embedding Projector's two-file TSV contract.

The Projector has no join column -- alignment is positional, so row i of
metadata.tsv describes the point at row i of vectors.tsv. The single rule that
prevents every alignment bug is: read ONE ordered clip list, then write both
files from that same list without re-sorting or re-filtering in between.
That is what this module does, and validate_export.py proves it afterwards.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.sources import sanitize


def main() -> int:
    ap = argparse.ArgumentParser(description="Export vectors.tsv + metadata.tsv.")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--splits", default="", help="comma list; default: all clips")
    ap.add_argument("--max-points", type=int, default=0,
                    help="0 = no cap; the hosted projector slows past roughly 50k")
    args = ap.parse_args()

    config.ensure_dirs()
    emb_path = config.EMB_DIR / f"embeddings_{args.tag}.npy"
    order_path = config.EMB_DIR / f"clip_order_{args.tag}.csv"
    if not emb_path.exists() or not order_path.exists():
        print(f"missing {emb_path.name} / {order_path.name} -- run embed.py --tag {args.tag}")
        return 1

    embeddings = np.load(emb_path)
    clips = pd.read_csv(order_path, dtype={"species_code": str}).fillna("")
    assert len(embeddings) == len(clips), \
        f"row mismatch before export: {len(embeddings)} vectors vs {len(clips)} clips"

    # Any filtering happens ONCE, as a positional index applied to BOTH arrays.
    keep = np.ones(len(clips), dtype=bool)
    if args.splits:
        wanted = {s.strip() for s in args.splits.split(",") if s.strip()}
        keep &= clips["split"].isin(wanted).to_numpy()
    idx = np.flatnonzero(keep)
    if args.max_points and len(idx) > args.max_points:
        rng = np.random.default_rng(config.SPLIT_SEED)
        idx = np.sort(rng.choice(idx, size=args.max_points, replace=False))

    embeddings = embeddings[idx]
    clips = clips.iloc[idx].reset_index(drop=True)

    out_dir = args.out_dir or (config.EXPORT_DIR / args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors_tsv = out_dir / "vectors.tsv"
    metadata_tsv = out_dir / "metadata.tsv"

    # --- vectors.tsv: no header, no id column, D floats per line ------------
    # newline="" stops Windows translating LF into CRLF: a trailing CR
    # would be glued onto the last value of every row and corrupt the parse.
    with open(vectors_tsv, "w", newline="", encoding="utf-8") as fh:
        np.savetxt(fh, embeddings, delimiter="\t", fmt="%.6f")

    # --- metadata.tsv: header row, same row order ---------------------------
    meta = pd.DataFrame(index=clips.index)
    for column in config.METADATA_COLUMNS:
        if column in clips.columns:
            meta[column] = clips[column].map(lambda v: sanitize(v, 90))
        else:
            meta[column] = ""
    meta.to_csv(metadata_tsv, sep="\t", index=False,
                lineterminator="\n", encoding="utf-8")

    # --- optional TensorBoard config (plan section 3.4) ---------------------
    cfg_blob = {
        "embeddings": [{
            "tensorName": f"bird_song_{args.tag}",
            "tensorShape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
            "tensorPath": "vectors.tsv",
            "metadataPath": "metadata.tsv",
        }]
    }
    (out_dir / "projector_config.json").write_text(json.dumps(cfg_blob, indent=2),
                                                   encoding="utf-8")

    print(f"tag         : {args.tag}")
    print(f"points      : {embeddings.shape[0]}")
    print(f"dimensions  : {embeddings.shape[1]}")
    print(f"species     : {clips['species_code'].nunique()}")
    print(f"sources     : {', '.join(sorted(clips['dataset_source'].unique()))}")
    print(f"\nwrote {vectors_tsv}  ({vectors_tsv.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {metadata_tsv}  ({metadata_tsv.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
