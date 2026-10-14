#!/usr/bin/env bash
# Three band-floor corpora side by side, so FINDINGS section 3 (the 1.5 kHz
# floor destroying both Columbidae) is settled with data before Phase 3b
# commits to one. Waits for the fetch to finish first.
set -uo pipefail
cd /workspace/birdclef
source /venv/main/bin/activate
export BIRDCLEF_ROOT=/workspace/birdclef OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
LOG=/workspace/birdclef/variants.log
: > "$LOG"

echo "waiting for FETCH_DONE..." >> "$LOG"
while [ ! -f /workspace/birdclef/FETCH_DONE ]; do sleep 30; done
echo "=== VARIANTS START $(date -u) ===" >> "$LOG"
echo "recordings in manifest: $(tail -n +2 data/interim/recordings.csv | wc -l)" >> "$LOG"

run () {   # tag, extra args...
  local tag=$1; shift
  local t0=$SECONDS
  echo "" >> "$LOG"
  echo "=== $tag :: $(date -u +%H:%M:%S) ===" >> "$LOG"
  python src/preprocess.py --tag "$tag" --workers 4 "$@" >> "$LOG" 2>&1
  echo "--- $tag exit=$? wall=$((SECONDS-t0))s ---" >> "$LOG"
}

run nofilter --min-bird-band 0.0
run band400  --band-floor 400
run band1500 --band-floor 1500

echo "" >> "$LOG"
echo "=== SURVIVAL BY VARIANT (FINDINGS section 3) ===" >> "$LOG"
python - >> "$LOG" 2>&1 <<'PY'
import pandas as pd
base = pd.read_csv("data/interim/recordings.csv")
per_species_recs = base.groupby("species_code").size()
rows = {}
for tag in ("nofilter", "band400", "band1500"):
    d = pd.read_csv(f"data/interim/clips_{tag}.csv")
    kept = d.groupby("species_code")["recording_id"].nunique()
    rows[tag] = (kept / per_species_recs * 100).round(1)
    print(f"{tag:<9} {len(d):>6} clips  {d['recording_id'].nunique():>5} recordings")
t = pd.DataFrame(rows).fillna(0.0)
print("\nrecording survival %, by species:")
print(t.sort_values("band1500").to_string())
print("\nthe two Columbidae:")
print(t.loc[[s for s in ("moudov", "whwdov") if s in t.index]].to_string())
PY
echo "=== VARIANTS DONE $(date -u) ===" >> "$LOG"
touch /workspace/birdclef/VARIANTS_DONE
