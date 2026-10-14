#!/usr/bin/env bash
# Resume from Phase 3a. Phases 1-2 already produced 12,007 clips on disk.
set -uo pipefail
APP=/opt/birdclef/app
PY=/opt/birdclef/venv/bin/python
LOG=/var/log/birdclef/pipeline3.log
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
die () { touch /var/log/birdclef/PIPELINE3_FAILED; exit 1; }

: > "$LOG"
rm -f /var/log/birdclef/PIPELINE3_DONE /var/log/birdclef/PIPELINE3_FAILED
rm -rf "$APP/data/embeddings/_shards/baseline"

run "0  spot-check clips"  $PY scripts/spot_check_clips.py --n 20 || true
run "3a embed (BEATs)"     $PY src/embed.py --tag baseline --batch-size 16 --threads 16 || die
run "4  export projector"  $PY src/export_projector.py --tag baseline || die
run "4b validate export"   $PY src/validate_export.py --tag baseline \
      --expect-order-from data/embeddings/clip_order_baseline.csv || die
run "5  analyse clusters"  $PY src/analyze_clusters.py --tag baseline || true
run "6  attribution"       $PY scripts/make_attribution.py || true

echo "" >> "$LOG"
echo "=== PIPELINE3 COMPLETE $(date -u) ===" >> "$LOG"
touch /var/log/birdclef/PIPELINE3_DONE
