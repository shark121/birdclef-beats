"""Central configuration: paths and the one audio contract everything obeys.

Phase 2 of the plan insists every source is normalised to a single audio
contract *before* it reaches the model. That contract lives here so the
fetchers, the preprocessor and the embedder cannot drift apart.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("BIRDCLEF_ROOT", Path(__file__).resolve().parents[1]))

CONFIG_DIR = ROOT / "config"
DATA_DIR = Path(os.environ.get("BIRDCLEF_DATA", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"          # as-downloaded audio, one dir per source
INTERIM_DIR = DATA_DIR / "interim"  # manifests
CLIPS_DIR = DATA_DIR / "clips"      # 16 kHz mono fixed-length segments
EMB_DIR = DATA_DIR / "embeddings"
EXPORT_DIR = Path(os.environ.get("BIRDCLEF_EXPORT", ROOT / "exports"))
CKPT_DIR = Path(os.environ.get("BIRDCLEF_CKPT", DATA_DIR / "checkpoints"))

# ---------------------------------------------------------------- audio contract
SAMPLE_RATE_HZ = 16_000   # BEATs operates on 16 kHz mono
CHANNELS = 1
CLIP_SECONDS = 5.0        # plan section 4: 5 s -> more points, slightly noisier
CLIP_HOP_SECONDS = 5.0    # non-overlapping segmentation
MIN_SOURCE_SECONDS = 3.0  # drop recordings too short to yield one full clip
MAX_SEGMENTS_PER_RECORDING = 8   # stop one long recording dominating a species
TRIM_TOP_DB = 35.0        # librosa silence trim aggressiveness
MIN_CLIP_RMS = 0.0035     # drop near-silent segments after trimming

# An RMS floor alone does not mean "there is a bird in this clip": wind, traffic
# and handling noise are LOUD and sail through it. A spot-check of the first
# corpus found 8 of 20 clips were sub-500 Hz rumble with no song at all. So also
# require a real share of the energy to sit in the passerine song band.
BIRD_BAND_HZ = (1500.0, 9000.0)
MIN_BIRD_BAND_FRACTION = 0.25   # set to 0.0 to disable the filter

# ---------------------------------------------------------------- splits
SPLIT_FRACTIONS = {"train": 0.7, "val": 0.15, "viz": 0.15}
SPLIT_SEED = 20260908

# ---------------------------------------------------------------- projector export
# Order is locked here and used by BOTH tsv writers, so row i of vectors.tsv and
# row i of metadata.tsv can never disagree (plan section 3.5).
METADATA_COLUMNS = [
    "clip_id",
    "species_code",
    "scientific_name",
    "common_name",
    "dataset_source",
    "xc_recording_id",
    "source_recording_id",
    "country",
    "location",
    "quality_rating",
    "recordist",
    "license",
    "duration_sec",
    "segment_start_sec",
    "sample_rate_hz",
    "split",
    "taxon_class",      # Aves / Amphibia / Insecta / ... ; empty for the 20-bird corpus
]

DATASET_SOURCES = ("xeno-canto", "birdclef", "inaturalist", "field")


# Which label space the pipeline runs on. species.json is the original 20 Gulf
# Coast birds; species_bc26.json is the 234-class BirdCLEF 2026 Pantanal set.
SPECIES_FILE = CONFIG_DIR / os.environ.get("BIRDCLEF_SPECIES_FILE", "species.json")


def load_species() -> list[dict]:
    with open(SPECIES_FILE, encoding="utf-8") as fh:
        return json.load(fh)["species"]


def target_per_species() -> int:
    with open(SPECIES_FILE, encoding="utf-8") as fh:
        return int(json.load(fh)["target_recordings_per_species"])


def ensure_dirs() -> None:
    for d in (RAW_DIR, INTERIM_DIR, CLIPS_DIR, EMB_DIR, EXPORT_DIR, CKPT_DIR):
        d.mkdir(parents=True, exist_ok=True)
