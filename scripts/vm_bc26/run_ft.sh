#!/usr/bin/env bash
set -euo pipefail
cd /workspace/bc26
source /venv/main/bin/activate
export BIRDCLEF_SPECIES_FILE=species_bc26.json
python src/finetune.py --tag bc26 --out-tag bc26_ft --epochs 10
python src/analyze_clusters.py --tag bc26_ft --k 10
echo FT_DONE
