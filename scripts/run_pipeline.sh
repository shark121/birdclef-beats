#!/usr/bin/env bash
# Phases 1 -> 5 in order. Each stage logs a marker so progress is pollable.
set -uo pipefail
APP=/opt/birdclef/app
PY=/opt/birdclef/venv/bin/python
LOG=/var/log/birdclef/pipeline.log
cd "$APP"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

stage () {
  echo ""                                   >> "$LOG"
  echo "=== STAGE $1 :: $(date -u +%H:%M:%S) ===" >> "$LOG"
}
run () {
  stage "$1"; shift
  "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "--- exit=$rc ---" >> "$LOG"
  return $rc
}

: > "$LOG"
rm -f /var/log/birdclef/PIPELINE_DONE /var/log/birdclef/PIPELINE_FAILED

run "1a fetch iNaturalist"  $PY src/fetch_inaturalist.py --per-species 60   || true
run "1b fetch Xeno-canto"   $PY src/fetch_xenocanto.py                      || true
run "1c ingest BirdCLEF"    $PY src/fetch_birdclef.py                       || true
run "1d ingest field"       $PY src/fetch_field.py                          || true
run "1e build manifest"     $PY src/build_manifest.py                       || \
  { touch /var/log/birdclef/PIPELINE_FAILED; exit 1; }
run "2  preprocess"         $PY src/preprocess.py --workers 16              || \
  { touch /var/log/birdclef/PIPELINE_FAILED; exit 1; }
run "3a embed (BEATs)"      $PY src/embed.py --tag baseline --batch-size 16 --threads 16 || \
  { touch /var/log/birdclef/PIPELINE_FAILED; exit 1; }
run "4  export projector"   $PY src/export_projector.py --tag baseline      || \
  { touch /var/log/birdclef/PIPELINE_FAILED; exit 1; }
run "4b validate export"    $PY src/validate_export.py --tag baseline \
      --expect-order-from data/embeddings/clip_order_baseline.csv           || \
  { touch /var/log/birdclef/PIPELINE_FAILED; exit 1; }

echo "" >> "$LOG"
echo "=== PIPELINE COMPLETE $(date -u) ===" >> "$LOG"
touch /var/log/birdclef/PIPELINE_DONE
