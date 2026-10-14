"""Prove the Phase 4 export contract without needing any audio or a model.

Builds a synthetic embedding set with deliberately hostile metadata (tabs,
newlines, unicode, empty fields), exports it, and runs the section 3.5
validator against it -- including the positional-identity check.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config

N, D = 250, 768
TAG = "smoketest"

config.ensure_dirs()
rng = np.random.default_rng(7)
species = [s["species_code"] for s in config.load_species()]

rows = []
for i in range(N):
    sp = species[i % len(species)]
    rows.append({
        "clip_id": f"inat{100000 + i}s{i}_seg{i % 8:02d}",
        "recording_id": f"inaturalist_inat{100000 + i}s{i}",
        "clip_path": f"viz/{sp}/x{i}.wav",
        "split": ["train", "val", "viz"][i % 3],
        "segment_start_sec": round((i % 8) * 5.0, 3),
        "duration_sec": 5.0,
        "sample_rate_hz": 16000,
        "species_code": sp,
        "scientific_name": "Genus species",
        "common_name": "Common\tName\nWith Junk" if i % 17 == 0 else "Common Name",
        "dataset_source": ["inaturalist", "xeno-canto", "birdclef", "field"][i % 4],
        "xc_recording_id": "" if i % 2 else str(500000 + i),
        "source_recording_id": f"src{i}",
        "country": "United States",
        "location": "Galveston Bay, TX \u2014 marsh edge" if i % 5 else "",
        "quality_rating": "A",
        "recordist": "J.\tDoe" if i % 11 == 0 else "A. Recordist",
        "license": "CC-BY-NC",
    })

clips = pd.DataFrame(rows).sort_values("clip_id").reset_index(drop=True)
embeddings = rng.normal(size=(N, D)).astype(np.float32)

np.save(config.EMB_DIR / f"embeddings_{TAG}.npy", embeddings)
clips.to_csv(config.EMB_DIR / f"clip_order_{TAG}.csv", index=False)
print(f"synthetic set: {embeddings.shape}, {len(clips)} clip rows\n")

py = sys.executable
for cmd in (
    [py, "src/export_projector.py", "--tag", TAG],
    [py, "src/validate_export.py", "--tag", TAG,
     "--expect-order-from", str(config.EMB_DIR / f"clip_order_{TAG}.csv")],
):
    print("$", " ".join(Path(c).name if c.endswith(".py") else c for c in cmd[1:]))
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print(proc.stderr.strip())
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    print()
print("Phase 4 export contract verified.")
