#!/usr/bin/env bash
# Enriched re-run: more recordings per species, and the ffmpeg decode fallback
# that recovers the .m4a files the first pass had to drop.
set -uo pipefail
APP=/opt/birdclef/app
PY=/opt/birdclef/venv/bin/python
LOG=/var/log/birdclef/pipeline2.log
cd "$APP"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

run () {
  local name="$1"; shift
  echo ""                                          >> "$LOG"
  echo "=== STAGE $name :: $(date -u +%H:%M:%S) ===" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "--- exit=$rc ---" >> "$LOG"
  return $rc
}
die () { touch /var/log/birdclef/PIPELINE2_FAILED; exit 1; }

: > "$LOG"
rm -f /var/log/birdclef/PIPELINE2_DONE /var/log/birdclef/PIPELINE2_FAILED

run "1a fetch iNaturalist (300/sp)" $PY src/fetch_inaturalist.py --per-species 300 || true
run "1b fetch Xeno-canto"           $PY src/fetch_xenocanto.py  || true
run "1c ingest BirdCLEF"            $PY src/fetch_birdclef.py   || true
run "1d ingest field"               $PY src/fetch_field.py      || true
run "1e build manifest"             $PY src/build_manifest.py   || die
run "2  preprocess"                 $PY src/preprocess.py --workers 16 || die
run "3a embed (BEATs)"              $PY src/embed.py --tag baseline --batch-size 16 --threads 16 || die
run "4  export projector"           $PY src/export_projector.py --tag baseline || die
run "4b validate export"            $PY src/validate_export.py --tag baseline \
      --expect-order-from data/embeddings/clip_order_baseline.csv || die
run "5  analyse clusters"           $PY src/analyze_clusters.py --tag baseline || true
run "6  attribution"                $PY scripts/make_attribution.py || true

echo "" >> "$LOG"
echo "=== PIPELINE2 COMPLETE $(date -u) ===" >> "$LOG"
touch /var/log/birdclef/PIPELINE2_DONE
