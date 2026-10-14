"""Phase 1 fetcher: BirdCLEF (Kaggle).

STATUS: wired and ready, awaiting one credential.

    1. https://www.kaggle.com/settings -> "Create New Token" -> kaggle.json
    2. place it at ~/.kaggle/kaggle.json (chmod 600)
    3. accept the competition rules on the competition's page
    4. python src/fetch_birdclef.py --competition birdclef-2026

The bundle is train_audio/<label>/*.ogg plus train.csv (train_metadata.csv
before 2025). The column set drifts year to year, so every column is looked up
through COLUMN_ALIASES rather than assumed; a missing optional column degrades
to an empty string instead of crashing the run.

2026 specifics: labels are eBird codes for birds but iNaturalist taxon ids for
everything else (frogs, insects, mammals), and `collection` says whether a
recording came from Xeno-canto or iNaturalist. That goes into dataset_source
(birdclef-xc / birdclef-inat) so the source batch-effect test has two sources.
Audio is symlinked into data/raw rather than copied -- the bundle is 11 GB.

Without credentials this exits 0 with an empty record set.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.sources import record, write_records

# BirdCLEF renames columns between years; accept any of these spellings.
COLUMN_ALIASES = {
    "species_code":    ("primary_label", "species_code", "ebird_code"),
    "scientific_name": ("scientific_name", "sci_name"),
    "common_name":     ("common_name", "species", "en"),
    "recordist":       ("author", "recordist", "rec"),
    "license":         ("license", "lic"),
    "quality_rating":  ("rating", "quality", "q"),
    "filename":        ("filename", "file", "path"),
    "latitude":        ("latitude", "lat"),
    "longitude":       ("longitude", "lng", "lon"),
    "url":             ("url", "source_url"),
    "taxon_class":     ("class_name",),
    "collection":      ("collection",),
}


def pick(row: dict, field: str) -> str:
    for alias in COLUMN_ALIASES.get(field, ()):
        if alias in row and row[alias] not in (None, ""):
            return str(row[alias])
    return ""


def xc_id_from(row: dict, filename: str) -> str:
    """BirdCLEF audio is largely re-hosted Xeno-canto; recover the XC id so
    build_manifest can dedupe it against a direct Xeno-canto pull."""
    url = pick(row, "url")
    if "xeno-canto.org" in url:
        tail = url.rstrip("/").split("/")[-1]
        digits = "".join(ch for ch in tail if ch.isdigit())
        if digits:
            return digits
    stem = Path(filename).stem
    if stem.upper().startswith("XC"):
        digits = "".join(ch for ch in stem[2:] if ch.isdigit())
        if digits:
            return digits
    return ""


def have_kaggle_credentials() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN") or (
            os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        return True
    home = Path.home() / ".kaggle"
    return (home / "kaggle.json").exists() or (home / "access_token").exists()


def try_download(competition: str, dest: Path) -> bool:
    if not have_kaggle_credentials():
        return False
    if shutil.which("kaggle") is None:
        print("kaggle CLI not installed:  pip install kaggle")
        return False
    dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading Kaggle competition '{competition}' -> {dest} (large)")
    proc = subprocess.run(
        ["kaggle", "competitions", "download", "-c", competition, "-p", str(dest)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        print("kaggle download failed:", proc.stderr.strip()[:400])
        return False
    for zip_path in dest.glob("*.zip"):
        shutil.unpack_archive(str(zip_path), str(dest))
        zip_path.unlink()
    return True


def find_metadata(bundle: Path) -> Path | None:
    for name in ("train.csv", "train_metadata.csv"):
        if (bundle / name).exists():
            return bundle / name
    return None


def ingest(bundle: Path, raw_root: Path) -> list[dict]:
    meta_csv = find_metadata(bundle)
    audio_root = bundle / "train_audio"
    if meta_csv is None or not audio_root.exists():
        print(f"no BirdCLEF bundle at {bundle} (need train.csv + train_audio/)")
        return []

    wanted = {s["species_code"]: s for s in config.load_species()}
    per_species_cap = config.target_per_species()
    counts: dict[str, int] = {}
    out: list[dict] = []

    with open(meta_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            code = pick(row, "species_code")
            if code not in wanted or counts.get(code, 0) >= per_species_cap:
                continue
            filename = pick(row, "filename")
            src = audio_root / filename
            if not src.exists():
                continue
            dest = raw_root / "birdclef" / code / Path(filename).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                try:
                    dest.symlink_to(src.resolve())
                except OSError:                  # e.g. Windows without symlink rights
                    shutil.copy2(src, dest)
            sp = wanted[code]
            collection = pick(row, "collection").lower()
            out.append(record(
                source_recording_id=f"birdclef_{Path(filename).stem}",
                dataset_source=f"birdclef-{collection}" if collection else "birdclef",
                taxon_class=pick(row, "taxon_class") or sp.get("taxon_class", ""),
                species_code=code,
                scientific_name=pick(row, "scientific_name") or sp["scientific_name"],
                common_name=pick(row, "common_name") or sp["common_name"],
                audio_path=str(dest.relative_to(raw_root)),
                xc_recording_id=xc_id_from(row, filename),
                country="",
                location=f"{pick(row,'latitude')},{pick(row,'longitude')}".strip(","),
                quality_rating=pick(row, "quality_rating"),
                recordist=pick(row, "recordist"),
                license=pick(row, "license"),
                source_url=pick(row, "url"),
            ))
            counts[code] = counts.get(code, 0) + 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest a BirdCLEF bundle.")
    ap.add_argument("--competition", default="birdclef-2026")
    ap.add_argument("--bundle", type=Path, default=None,
                    help="existing extracted bundle dir (skips download)")
    args = ap.parse_args()

    config.ensure_dirs()
    out_path = config.INTERIM_DIR / "records_birdclef.jsonl"
    bundle = args.bundle or (config.RAW_DIR / "_birdclef_bundle")

    if find_metadata(bundle) is None:
        if not try_download(args.competition, bundle):
            print("BirdCLEF skipped -- no Kaggle credentials (see module docstring).")
            write_records([], out_path)
            return 0

    records = ingest(bundle, config.RAW_DIR)
    n = write_records(records, out_path)
    print(f"wrote {n} recording records -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
