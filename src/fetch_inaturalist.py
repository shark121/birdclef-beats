"""Phase 1 fetcher: iNaturalist research-grade audio observations.

Why this source exists in the plan's stead: Xeno-canto retired API v2 and v3
now requires a per-account key (see fetch_xenocanto.py). iNaturalist's v1 API
is open, needs no credential, and carries every field metadata.tsv wants --
species, recordist, licence, place and coordinates -- so the pipeline can be
proven end-to-end today and Xeno-canto simply adds rows later.

Idempotent: an already-downloaded sound file is never re-fetched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.sources import download, get_json, record, write_records

API = "https://api.inaturalist.org/v1"
# Licences we are willing to redistribute/attribute. Anything else (including
# "all rights reserved", which iNat reports as a null license_code) is skipped.
OK_LICENSES = {"cc0", "cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa"}
POLITE_PAUSE = 1.1          # iNat asks for <= ~1 request/second sustained
TAXON_CACHE = "inat_taxon_ids.json"


def resolve_taxon_id(scientific_name: str, cache: dict) -> int | None:
    """Look up an iNat taxon id by scientific name, memoised to disk."""
    if scientific_name in cache:
        return cache[scientific_name]
    data = get_json(f"{API}/taxa", {"q": scientific_name, "rank": "species"})
    taxon_id = None
    for result in data.get("results", []):
        if result.get("name", "").lower() == scientific_name.lower():
            taxon_id = result["id"]
            break
    cache[scientific_name] = taxon_id
    time.sleep(POLITE_PAUSE)
    return taxon_id


def country_of(place_guess: str) -> str:
    """iNat gives a free-text place string; the trailing token is the country."""
    if not place_guess:
        return ""
    return place_guess.split(",")[-1].strip()


def harvest_species(sp: dict, taxon_id: int, want: int, raw_root: Path) -> list[dict]:
    """Page the observations endpoint until we have `want` usable recordings."""
    out: list[dict] = []
    page = 1
    seen_pages_without_gain = 0
    while len(out) < want and page <= 25:
        data = get_json(
            f"{API}/observations",
            {
                "taxon_id": taxon_id,
                "sounds": "true",
                "quality_grade": "research",
                "per_page": 100,
                "page": page,
                "order_by": "votes",   # community-favoured observations first
                "order": "desc",
            },
        )
        results = data.get("results", [])
        if not results:
            break
        before = len(out)
        for obs in results:
            if len(out) >= want:
                break
            for snd in obs.get("sounds") or []:
                url = snd.get("file_url")
                lic = (snd.get("license_code") or "").lower()
                if not url or lic not in OK_LICENSES:
                    continue
                ext = Path(url.split("?")[0]).suffix.lower() or ".mp3"
                if ext not in (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mpga", ".aac"):
                    continue
                rid = f"inat{obs['id']}s{snd.get('id', 0)}"
                dest = raw_root / "inaturalist" / sp["species_code"] / f"{rid}{ext}"
                if not download(url, dest):
                    continue
                user = obs.get("user") or {}
                place = obs.get("place_guess") or ""
                out.append(record(
                    source_recording_id=rid,
                    dataset_source="inaturalist",
                    species_code=sp["species_code"],
                    scientific_name=sp["scientific_name"],
                    common_name=sp["common_name"],
                    audio_path=str(dest.relative_to(raw_root)),
                    country=country_of(place),
                    location=place,
                    quality_rating=obs.get("quality_grade", ""),
                    recordist=user.get("name") or user.get("login") or "",
                    license=lic.upper(),
                    source_url=f"https://www.inaturalist.org/observations/{obs['id']}",
                ))
                break          # one sound per observation keeps species balanced
        if len(out) == before:
            seen_pages_without_gain += 1
            if seen_pages_without_gain >= 3:
                break
        else:
            seen_pages_without_gain = 0
        page += 1
        time.sleep(POLITE_PAUSE)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch iNaturalist bird audio.")
    ap.add_argument("--per-species", type=int, default=None,
                    help="override config target_recordings_per_species")
    ap.add_argument("--limit-species", type=int, default=None)
    args = ap.parse_args()

    config.ensure_dirs()
    species = config.load_species()
    if args.limit_species:
        species = species[: args.limit_species]
    want = args.per_species or config.target_per_species()

    cache_path = config.INTERIM_DIR / TAXON_CACHE
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    all_records: list[dict] = []
    for i, sp in enumerate(species, 1):
        taxon_id = resolve_taxon_id(sp["scientific_name"], cache)
        cache_path.write_text(json.dumps(cache, indent=2))
        if taxon_id is None:
            print(f"[{i:>2}/{len(species)}] {sp['species_code']:<7} NO TAXON MATCH", flush=True)
            continue
        got = harvest_species(sp, taxon_id, want, config.RAW_DIR)
        all_records.extend(got)
        print(f"[{i:>2}/{len(species)}] {sp['species_code']:<7} taxon={taxon_id:<8} "
              f"recordings={len(got)}", flush=True)

    out = config.INTERIM_DIR / "records_inaturalist.jsonl"
    n = write_records(all_records, out)
    print(f"\nwrote {n} recording records -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
