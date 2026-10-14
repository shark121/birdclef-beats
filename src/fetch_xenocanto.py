"""Phase 1 fetcher: Xeno-canto, API v3.

STATUS: wired and ready, awaiting one credential.

Xeno-canto retired API v2 (it now returns HTTP 404 with an explanatory body)
and v3 requires a free per-account key:

    1. sign in / register at https://xeno-canto.org
    2. copy the key from https://xeno-canto.org/account
    3. export XC_API_KEY=<key>          (or drop it in config/xc_api_key.txt)
    4. python src/fetch_xenocanto.py

Without the key this exits 0 and writes an empty record set, so the rest of
the pipeline still runs -- Xeno-canto rows simply appear on the next run.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.sources import download, get_json, record, write_records

API = "https://xeno-canto.org/api/3/recordings"
POLITE_PAUSE = 1.0


def api_key() -> str | None:
    key = os.environ.get("XC_API_KEY", "").strip()
    if key:
        return key
    key_file = config.CONFIG_DIR / "xc_api_key.txt"
    if key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()
        if key:
            return key
    return None


def audio_url(rec: dict) -> str:
    """v3 exposes a direct file URL; fall back to the canonical download path."""
    url = (rec.get("file") or "").strip()
    if url.startswith("//"):
        url = "https:" + url
    if not url:
        url = f"https://xeno-canto.org/{rec.get('id')}/download"
    return url


def harvest_species(sp: dict, key: str, want: int, raw_root: Path,
                    quality: str) -> list[dict]:
    out: list[dict] = []
    page = 1
    while len(out) < want and page <= 15:
        genus, _, epithet = sp["scientific_name"].partition(" ")
        query = f'gen:"{genus}" sp:"{epithet}" q:"{quality}"'
        data = get_json(API, {"query": query, "key": key, "page": page})
        recs = data.get("recordings", [])
        if not recs:
            break
        for rec in recs:
            if len(out) >= want:
                break
            xc_id = str(rec.get("id", "")).strip()
            if not xc_id:
                continue
            url = audio_url(rec)
            ext = Path(url.split("?")[0]).suffix.lower()
            if ext not in (".mp3", ".wav", ".ogg", ".flac", ".m4a"):
                ext = ".mp3"
            dest = raw_root / "xeno-canto" / sp["species_code"] / f"xc{xc_id}{ext}"
            if not download(url, dest):
                continue
            out.append(record(
                source_recording_id=f"xc{xc_id}",
                dataset_source="xeno-canto",
                species_code=sp["species_code"],
                scientific_name=f"{rec.get('gen','')} {rec.get('sp','')}".strip()
                                 or sp["scientific_name"],
                common_name=rec.get("en") or sp["common_name"],
                audio_path=str(dest.relative_to(raw_root)),
                xc_recording_id=xc_id,
                country=rec.get("cnt", ""),
                location=rec.get("loc", ""),
                quality_rating=rec.get("q", ""),
                recordist=rec.get("rec", ""),
                license=(rec.get("lic") or "").replace("//creativecommons.org/licenses/", "CC-")
                         .replace("/", "-").strip("-").upper(),
                source_url=f"https://xeno-canto.org/{xc_id}",
            ))
        total_pages = int(data.get("numPages", 1) or 1)
        if page >= total_pages:
            break
        page += 1
        time.sleep(POLITE_PAUSE)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Xeno-canto audio (API v3).")
    ap.add_argument("--per-species", type=int, default=None)
    ap.add_argument("--quality", default="A", help='XC quality grade, e.g. "A"')
    args = ap.parse_args()

    config.ensure_dirs()
    out_path = config.INTERIM_DIR / "records_xeno-canto.jsonl"

    key = api_key()
    if not key:
        print("XC_API_KEY not set -- skipping Xeno-canto (see module docstring).")
        print("  get a key: https://xeno-canto.org/account")
        write_records([], out_path)
        return 0

    species = config.load_species()
    want = args.per_species or config.target_per_species()
    all_records: list[dict] = []
    for i, sp in enumerate(species, 1):
        got = harvest_species(sp, key, want, config.RAW_DIR, args.quality)
        all_records.extend(got)
        print(f"[{i:>2}/{len(species)}] {sp['species_code']:<7} recordings={len(got)}",
              flush=True)

    n = write_records(all_records, out_path)
    print(f"\nwrote {n} recording records -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
