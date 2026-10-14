"""Phase 6: per-recording attribution list, grouped by licence.

Every source in this corpus is used under a licence that requires crediting the
recordist. This regenerates that credit list from the manifest so it can never
drift from what was actually embedded.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config

man = config.INTERIM_DIR / "recordings.csv"
if not man.exists():
    raise SystemExit(f"missing {man} -- run build_manifest.py first")

df = pd.read_csv(man).fillna("")
out = config.EXPORT_DIR / "ATTRIBUTION.md"
out.parent.mkdir(parents=True, exist_ok=True)

lines = [
    "# Attribution",
    "",
    "Every recording embedded in this project, credited to its recordist and",
    "licence. Generated from `data/interim/recordings.csv`; regenerate with",
    "`python scripts/make_attribution.py`.",
    "",
    f"Total recordings: **{len(df):,}**",
    "",
]

by_source = df.groupby("dataset_source")
lines += ["## Summary by source and licence", "", "| source | licence | recordings |", "|---|---|---|"]
for source, grp in by_source:
    for lic, sub in grp.groupby("license"):
        lines.append(f"| {source} | {lic or '(unspecified)'} | {len(sub)} |")
lines.append("")

for source, grp in by_source:
    lines += [f"## {source} ({len(grp):,} recordings)", ""]
    by_lic: dict[str, list[str]] = defaultdict(list)
    for _, row in grp.sort_values(["species_code", "source_recording_id"]).iterrows():
        entry = (f"{row['source_recording_id']} -- {row['common_name'] or row['species_code']}"
                 f" -- {row['recordist'] or 'unknown recordist'}")
        if row["source_url"]:
            entry += f" -- <{row['source_url']}>"
        by_lic[row["license"] or "(unspecified)"].append(entry)
    for lic in sorted(by_lic):
        lines += [f"### {lic} ({len(by_lic[lic])})", ""]
        lines += [f"- {e}" for e in by_lic[lic]]
        lines.append("")

out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print(f"wrote {out}  ({len(df)} recordings, {df['license'].nunique()} distinct licences)")
