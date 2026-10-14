#!/usr/bin/env bash
# BirdCLEF 2026: preprocess -> embed -> analyse -> export, all logged.
set -euo pipefail
cd /workspace/bc26
source /venv/main/bin/activate
export BIRDCLEF_SPECIES_FILE=species_bc26.json
python src/preprocess.py --tag bc26 --min-bird-band 0 --pad-short --min-source-seconds 1 --workers 80
python src/embed.py --corpus bc26 --tag bc26 --device cuda --batch-size 128
python src/analyze_clusters.py --tag bc26 --k 10
echo PIPELINE_DONE
