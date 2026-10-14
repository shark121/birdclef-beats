"""Shared plumbing for every Phase 1 fetcher.

All fetchers emit the SAME per-recording record shape, so build_manifest can
concatenate them without special-casing a source. Anything source-specific
(API paging, licence vocabularies) is confined to the fetcher itself.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import requests

UA = "birdclef2.0-embedding-viz/1.0 (research; contact via project maintainer)"

# The projector's TSV contract forbids literal tabs/newlines inside any field
# (plan section 3.5). Sanitise once, here, so no writer has to remember.
_WS = re.compile(r"[\t\r\n]+")


def sanitize(value: Any, max_len: int = 120) -> str:
    """Make a value safe for a TSV cell: no tabs, no newlines, bounded length."""
    if value is None:
        return ""
    text = _WS.sub(" ", str(value)).strip()
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "\u2026"
    return text


def record(
    *,
    source_recording_id: str,
    dataset_source: str,
    species_code: str,
    scientific_name: str,
    common_name: str,
    audio_path: str,
    xc_recording_id: str = "",
    country: str = "",
    location: str = "",
    quality_rating: str = "",
    recordist: str = "",
    license: str = "",
    source_url: str = "",
    taxon_class: str = "",
) -> dict:
    """The one canonical per-recording record shape."""
    return {
        "taxon_class": sanitize(taxon_class, 16),
        "source_recording_id": sanitize(source_recording_id, 80),
        "dataset_source": sanitize(dataset_source, 24),
        "species_code": sanitize(species_code, 16),
        "scientific_name": sanitize(scientific_name, 80),
        "common_name": sanitize(common_name, 80),
        "xc_recording_id": sanitize(xc_recording_id, 24),
        "country": sanitize(country, 60),
        "location": sanitize(location, 90),
        "quality_rating": sanitize(quality_rating, 8),
        "recordist": sanitize(recordist, 60),
        "license": sanitize(license, 40),
        "audio_path": str(audio_path),
        "source_url": sanitize(source_url, 200),
    }


def get_json(url: str, params: dict | None = None, *, retries: int = 4,
             timeout: int = 60, pause: float = 1.0) -> dict:
    """GET returning JSON, with backoff. Raises on final failure."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout,
                                headers={"User-Agent": UA})
            if resp.status_code == 429:            # rate limited -> back off hard
                time.sleep(pause * (attempt + 1) * 5)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:                    # noqa: BLE001 - retried below
            last = exc
            time.sleep(pause * (2 ** attempt))
    raise RuntimeError(f"GET failed after {retries} tries: {url} :: {last}")


def download(url: str, dest: Path, *, retries: int = 3, timeout: int = 120,
             min_bytes: int = 2048) -> bool:
    """Stream a file to dest. Idempotent: an existing non-trivial file is kept.

    Returns True if dest holds a usable file when we return.
    """
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return True                                  # already have it -> skip
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout,
                              headers={"User-Agent": UA}) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fh.write(chunk)
            if tmp.stat().st_size < min_bytes:
                tmp.unlink(missing_ok=True)
                return False
            tmp.replace(dest)
            return True
        except Exception:                            # noqa: BLE001 - retried
            tmp.unlink(missing_ok=True)
            time.sleep(1.5 * (attempt + 1))
    return False


def write_records(records: Iterable[dict], path: Path) -> int:
    """Persist a fetcher's records as JSONL. Returns the count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
