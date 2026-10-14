#!/usr/bin/env bash
# Variant run: same corpus, same model, but Phase 2 now also requires a real
# share of energy in the 1.5-9 kHz bird band. Exported under tag "filtered" so
# it sits alongside "baseline" in the Projector for a like-for-like comparison.
set -uo pipefail
APP=/opt/birdclef/app
PY=/opt/birdclef/venv/bin/python
LOG=/var/log/birdclef/filtered.log
cd "$APP"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

run () {
  local name="$1"; shift
  echo ""                                            >> "$LOG"
  echo "=== STAGE $name :: $(date -u +%H:%M:%S) ===" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "--- exit=$rc ---" >> "$LOG"
  return $rc
}
die () { touch /var/log/birdclef/FILTERED_FAILED; exit 1; }

: > "$LOG"
rm -f /var/log/birdclef/FILTERED_DONE /var/log/birdclef/FILTERED_FAILED

run "2  preprocess (band filter)" $PY src/preprocess.py --workers 16 --min-bird-band 0.25 || die
run "2b spot-check clips"         $PY scripts/spot_check_clips.py --n 20 || true
run "3a embed (BEATs)"            $PY src/embed.py --tag filtered --batch-size 16 --threads 16 || die
run "4  export projector"         $PY src/export_projector.py --tag filtered || die
run "4b validate export"          $PY src/validate_export.py --tag filtered \
      --expect-order-from data/embeddings/clip_order_filtered.csv || die
run "5  analyse clusters"         $PY src/analyze_clusters.py --tag filtered || true

echo "" >> "$LOG"
echo "=== FILTERED COMPLETE $(date -u) ===" >> "$LOG"
touch /var/log/birdclef/FILTERED_DONE
