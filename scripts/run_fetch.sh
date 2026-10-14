#!/usr/bin/env bash
# Full corpus fetch on this box. Stops before preprocess: the band-floor
# decision (FINDINGS section 3) is still open.
set -uo pipefail
cd /workspace/birdclef
source /venv/main/bin/activate
export BIRDCLEF_ROOT=/workspace/birdclef OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
LOG=/workspace/birdclef/fetch.log
: > "$LOG"
echo "=== FETCH START $(date -u) ===" >> "$LOG"
python src/fetch_inaturalist.py --per-species 300 >> "$LOG" 2>&1
echo "--- fetch exit=$? $(date -u) ---" >> "$LOG"
python src/build_manifest.py >> "$LOG" 2>&1
echo "--- manifest exit=$? ---" >> "$LOG"
echo "raw files: $(find data/raw -type f | wc -l)" >> "$LOG"
du -sh data/raw >> "$LOG" 2>&1
echo "=== FETCH DONE $(date -u) ===" >> "$LOG"
touch /workspace/birdclef/FETCH_DONE
