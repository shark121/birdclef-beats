"""Phase 1 final step: merge every source into one recording-level manifest.

The plan flags the important overlap: BirdCLEF's training audio is itself
largely re-hosted Xeno-canto, so the same recording can arrive twice and would
otherwise become two points in the projector. We dedupe on the original
Xeno-canto recording id and keep the most canonical copy.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.sources import read_records

# Lower number wins when the same XC recording shows up in two sources.
# Xeno-canto is the original publisher; BirdCLEF re-hosts it; field audio is
# unique by definition and can never be deduped away.
SOURCE_PRIORITY = {"field": 0, "xeno-canto": 1, "birdclef": 2, "inaturalist": 3}


def main() -> int:
    config.ensure_dirs()
    frames = []
    for source in config.DATASET_SOURCES:
        path = config.INTERIM_DIR / f"records_{source}.jsonl"
        recs = read_records(path)
        print(f"  {source:<12} {len(recs):>5} records  ({path.name})")
        if recs:
            frames.append(pd.DataFrame(recs))

    if not frames:
        print("\nNo records from any source. Run the fetchers first.")
        return 1

    df = pd.concat(frames, ignore_index=True)
    before = len(df)

    # 1) exact duplicates of the same source recording
    df = df.drop_duplicates(subset=["dataset_source", "source_recording_id"])

    # 2) cross-source duplicates sharing an original Xeno-canto id
    df["_prio"] = df["dataset_source"].map(SOURCE_PRIORITY).fillna(9).astype(int)
    has_xc = df["xc_recording_id"].astype(str).str.len() > 0
    xc_part = (df[has_xc]
               .sort_values("_prio")
               .drop_duplicates(subset=["xc_recording_id"], keep="first"))
    df = pd.concat([df[~has_xc], xc_part], ignore_index=True)
    df = df.drop(columns=["_prio"]).sort_values(
        ["species_code", "dataset_source", "source_recording_id"]).reset_index(drop=True)

    # a stable, globally-unique recording key used downstream as the split unit
    df["recording_id"] = df["dataset_source"].str.replace("-", "", regex=False) \
                         + "_" + df["source_recording_id"].astype(str)

    out = config.INTERIM_DIR / "recordings.csv"
    df.to_csv(out, index=False)

    print(f"\nmerged {before} -> {len(df)} recordings after dedupe "
          f"({before - len(df)} removed)")
    print("\nby source:")
    for src, n in Counter(df["dataset_source"]).most_common():
        print(f"  {src:<12} {n:>5}")
    print(f"\nspecies represented: {df['species_code'].nunique()} / {len(config.load_species())}")
    thin = df.groupby("species_code").size().sort_values()
    print("thinnest species:", ", ".join(f"{k}={v}" for k, v in thin.head(5).items()))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
