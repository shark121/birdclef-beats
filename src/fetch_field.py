"""Phase 1 fetcher: original Galveston Bay / Houston-area field recordings.

STATUS: path built and tested, awaiting audio.

Field recordings carry no external metadata record, so the plan calls for the
same fields to be captured by hand at collection time. That hand-entry lives in
config/field_recordings.csv; this module joins it to the audio and emits records
identical in shape to every other source, giving dataset_source="field" as a
fourth value to check for a site-specific batch effect in the projection.

To add recordings:
    1. drop audio under  data/raw/field/<species_code>/<file>
       (use species_code "unknown" for anything not yet identified)
    2. add one row per file to config/field_recordings.csv
    3. re-run the pipeline -- nothing else changes
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.sources import record, write_records

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff")
TEMPLATE_HEADER = [
    "filename", "species_code", "scientific_name", "common_name",
    "date", "location", "latitude", "longitude",
    "equipment", "recordist", "license", "notes",
]


def ensure_template() -> Path:
    """Create the hand-entry sheet on first run so the schema is discoverable."""
    path = config.CONFIG_DIR / "field_recordings.csv"
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(TEMPLATE_HEADER)
            writer.writerow([
                "# example - delete this row", "norcar", "Cardinalis cardinalis",
                "Northern Cardinal", "2026-09-08", "Galveston Bay, TX",
                "29.5000", "-94.9000", "Zoom H5 + Rode NTG2",
                "S. Boakye", "CC-BY-4.0", "dawn chorus, light wind",
            ])
    return path


def main() -> int:
    config.ensure_dirs()
    sheet = ensure_template()
    field_root = config.RAW_DIR / "field"
    field_root.mkdir(parents=True, exist_ok=True)
    out_path = config.INTERIM_DIR / "records_field.jsonl"

    by_name: dict[str, dict] = {}
    with open(sheet, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("filename") or "").strip()
            if name and not name.startswith("#"):
                by_name[name] = row

    audio_files = [p for p in field_root.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    if not audio_files:
        print(f"no field audio yet under {field_root}")
        print(f"  drop files there and describe them in {sheet}")
        write_records([], out_path)
        return 0

    known = {s["species_code"]: s for s in config.load_species()}
    records, undescribed = [], []
    for path in audio_files:
        meta = by_name.get(path.name)
        if meta is None:
            undescribed.append(path.name)
            continue
        code = (meta.get("species_code") or path.parent.name or "unknown").strip()
        sp = known.get(code, {})
        lat, lon = meta.get("latitude", ""), meta.get("longitude", "")
        loc = meta.get("location") or ""
        records.append(record(
            source_recording_id=f"field_{path.stem}",
            dataset_source="field",
            species_code=code,
            scientific_name=meta.get("scientific_name") or sp.get("scientific_name", ""),
            common_name=meta.get("common_name") or sp.get("common_name", ""),
            audio_path=str(path.relative_to(config.RAW_DIR)),
            country="United States",
            location=f"{loc} ({lat},{lon})".strip() if lat or lon else loc,
            quality_rating="field",
            recordist=meta.get("recordist", ""),
            license=meta.get("license", ""),
            source_url="",
        ))

    if undescribed:
        print(f"WARNING: {len(undescribed)} field file(s) have no row in {sheet.name} "
              f"and were skipped: {', '.join(undescribed[:5])}"
              + (" ..." if len(undescribed) > 5 else ""))

    n = write_records(records, out_path)
    print(f"wrote {n} recording records -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
