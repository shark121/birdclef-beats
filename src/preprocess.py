"""Phase 2: normalise every recording to one audio contract, then segment.

Guarantees on the output (each is a plan acceptance criterion):
  * every clip is 16 kHz mono                      -> resample + mono-mix
  * near-silent clips are gone                     -> trim + RMS floor
  * every clip has identical duration               -> fixed-length windows
  * segment_start_sec is recorded per segment       -> offset kept in the manifest
  * no recording spans two splits                   -> split assigned per recording

Clips are written as 16-bit WAV rather than a packed array so the "spot-check
20 random clips" criterion is something you can actually listen to.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def assign_split(recording_id: str) -> str:
    """Deterministic, recording-level split. Same id -> same split, always,
    so re-running preprocessing never leaks a recording across splits."""
    digest = hashlib.sha1(f"{config.SPLIT_SEED}:{recording_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    cum = 0.0
    for name, frac in config.SPLIT_FRACTIONS.items():
        cum += frac
        if bucket < cum:
            return name
    return "viz"


def load_audio(path: Path, sr: int) -> np.ndarray:
    """Load any source file as mono float32 at `sr`.

    libsndfile has no AAC/M4A decoder and librosa 0.11 dropped the old audioread
    fallback, so a plain librosa.load() rejects every .m4a file iNaturalist
    serves -- which silently cost us two thirds of the corpus. Fall back to
    ffmpeg, which decodes those fine.
    """
    try:
        audio, _ = librosa.load(str(path), sr=sr, mono=True)
        if audio.size:
            return audio
    except Exception:
        pass
    proc = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", str(path), "-f", "f32le",
         "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"both librosa and ffmpeg failed to decode {path.name}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def bird_band_fraction(segment: np.ndarray, sr: int,
                       band: tuple[float, float]) -> float:
    """Share of spectral energy inside the passerine song band."""
    spec = np.abs(np.fft.rfft(segment * np.hanning(len(segment)))) ** 2
    freqs = np.fft.rfftfreq(len(segment), 1.0 / sr)
    lo, hi = band
    total = spec.sum()
    if total <= 0:
        return 0.0
    return float(spec[(freqs >= lo) & (freqs <= hi)].sum() / total)


def process_recording(row: dict, min_band: float, band: tuple[float, float],
                      clips_dir: Path, min_source_s: float = config.MIN_SOURCE_SECONDS,
                      pad_short: bool = False) -> tuple[list[dict], str | None]:
    """Load one recording, emit its fixed-length clips. Never raises.

    pad_short: zero-pad a recording shorter than one clip up to clip length
    instead of dropping it. Without it, anything under 5 s after trimming yields
    nothing -- fine for the 20-bird corpus, but in BirdCLEF 2026 it would erase
    7% of recordings, mostly short insect and frog calls.
    """
    sr = config.SAMPLE_RATE_HZ
    win = int(config.CLIP_SECONDS * sr)
    hop = int(config.CLIP_HOP_SECONDS * sr)
    src = config.RAW_DIR / row["audio_path"]
    try:
        audio = load_audio(src, sr)
    except Exception as exc:                                   # noqa: BLE001
        return [], f"{row['recording_id']}: load failed: {type(exc).__name__}: {exc}"

    if audio.size == 0 or not np.isfinite(audio).all():
        return [], f"{row['recording_id']}: empty or non-finite audio"

    # trim leading/trailing silence before measuring duration
    audio, _ = librosa.effects.trim(audio, top_db=config.TRIM_TOP_DB)
    if audio.size < min_source_s * sr:
        return [], f"{row['recording_id']}: shorter than {min_source_s}s after trim"
    if pad_short and audio.size < win:
        audio = np.pad(audio, (0, win - audio.size))

    split = assign_split(row["recording_id"])
    out_dir = clips_dir / split / row["species_code"]
    out_dir.mkdir(parents=True, exist_ok=True)

    clips, kept = [], 0
    for seg_index, start in enumerate(range(0, max(1, audio.size - win + 1), hop)):
        if kept >= config.MAX_SEGMENTS_PER_RECORDING:
            break
        seg = audio[start:start + win]
        if seg.size < win:
            break                                    # only emit full-length windows
        if float(np.sqrt(np.mean(seg ** 2))) < config.MIN_CLIP_RMS:
            continue                                 # near-silent window
        band_frac = bird_band_fraction(seg, sr, band)
        if band_frac < min_band:
            continue                                 # loud, but it is not a bird
        peak = float(np.max(np.abs(seg)))
        if peak > 0:
            seg = seg / peak * 0.95                  # peak-normalise for consistency
        clip_id = f"{row['recording_id']}_seg{seg_index:02d}"
        dest = out_dir / f"{clip_id}.wav"
        try:
            sf.write(str(dest), seg.astype(np.float32), sr, subtype="PCM_16")
        except Exception as exc:                     # noqa: BLE001
            return clips, f"{row['recording_id']}: write failed: {exc}"
        clips.append({
            "clip_id": clip_id,
            "recording_id": row["recording_id"],
            "clip_path": str(dest.relative_to(config.CLIPS_DIR)),
            "split": split,
            "segment_start_sec": round(start / sr, 3),
            "bird_band_fraction": round(band_frac, 4),
            "duration_sec": round(win / sr, 3),
            "sample_rate_hz": sr,
            "species_code": row["species_code"],
            "scientific_name": row["scientific_name"],
            "common_name": row["common_name"],
            "dataset_source": row["dataset_source"],
            "xc_recording_id": row.get("xc_recording_id", ""),
            "source_recording_id": row["source_recording_id"],
            "country": row.get("country", ""),
            "location": row.get("location", ""),
            "quality_rating": row.get("quality_rating", ""),
            "recordist": row.get("recordist", ""),
            "license": row.get("license", ""),
            "taxon_class": row.get("taxon_class", ""),
        })
        kept += 1
    if not clips:
        return [], f"{row['recording_id']}: no window passed the RMS + bird-band filters"
    return clips, None


def main() -> int:
    ap = argparse.ArgumentParser(description="Preprocess raw audio into clips.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-bird-band", type=float,
                    default=config.MIN_BIRD_BAND_FRACTION,
                    help="min share of energy in the bird band; 0 disables the filter")
    ap.add_argument("--tag", default=None,
                    help="write clips under data/clips/<tag>/ and clips_<tag>.csv, "
                         "so band variants sit side by side instead of overwriting")
    ap.add_argument("--band-floor", type=float, default=config.BIRD_BAND_HZ[0],
                    help="low edge of the bird band in Hz. FINDINGS section 3: the "
                         "1500 default discards both Columbidae, ~400 keeps their coo")
    ap.add_argument("--min-source-seconds", type=float, default=config.MIN_SOURCE_SECONDS,
                    help="drop recordings shorter than this after trimming")
    ap.add_argument("--pad-short", action="store_true",
                    help="zero-pad recordings shorter than one clip instead of dropping them")
    args = ap.parse_args()

    band = (args.band_floor, config.BIRD_BAND_HZ[1])
    clips_dir = config.CLIPS_DIR / args.tag if args.tag else config.CLIPS_DIR
    suffix = f"_{args.tag}" if args.tag else ""

    config.ensure_dirs()
    man_path = config.INTERIM_DIR / "recordings.csv"
    if not man_path.exists():
        print(f"missing {man_path} -- run build_manifest.py first")
        return 1

    df = pd.read_csv(man_path).fillna("")
    if args.limit:
        df = df.head(args.limit)
    rows = df.to_dict("records")
    print(f"preprocessing {len(rows)} recordings with {args.workers} workers "
          f"-> {config.CLIP_SECONDS}s @ {config.SAMPLE_RATE_HZ}Hz mono")
    print(f"bird-band filter: >= {args.min_bird_band:.0%} of energy in "
          f"{band[0]/1000:.1f}-{band[1]/1000:.0f} kHz"
          + (f"   [tag: {args.tag}]" if args.tag else ""))

    all_clips: list[dict] = []
    problems: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_recording, r, args.min_bird_band, band,
                               clips_dir, args.min_source_seconds, args.pad_short):
                   r["recording_id"] for r in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                clips, problem = fut.result()
            except Exception:                        # noqa: BLE001
                clips, problem = [], f"{futures[fut]}: {traceback.format_exc(limit=1)}"
            all_clips.extend(clips)
            if problem:
                problems.append(problem)
            if i % 100 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)} recordings -> {len(all_clips)} clips", flush=True)

    if not all_clips:
        print("\nNo clips produced. See problems below.")
        for p in problems[:20]:
            print("  -", p)
        return 1

    clips_df = pd.DataFrame(all_clips).sort_values("clip_id").reset_index(drop=True)
    out = config.INTERIM_DIR / f"clips{suffix}.csv"
    clips_df.to_csv(out, index=False)

    print(f"\n{len(clips_df)} clips from {clips_df['recording_id'].nunique()} recordings")
    print("\nby split:")
    for split, n in clips_df["split"].value_counts().items():
        n_rec = clips_df.loc[clips_df["split"] == split, "recording_id"].nunique()
        print(f"  {split:<6} {n:>6} clips  ({n_rec} recordings)")

    # the no-leak criterion, asserted rather than assumed
    leak = clips_df.groupby("recording_id")["split"].nunique()
    assert (leak == 1).all(), f"split leak: {leak[leak > 1].index.tolist()[:5]}"
    print("\nsplit integrity: OK (no recording spans two splits)")

    if problems:
        log = config.INTERIM_DIR / f"preprocess_problems{suffix}.log"
        log.write_text("\n".join(problems), encoding="utf-8")
        print(f"\n{len(problems)} recordings skipped -> {log}")
        for p in problems[:5]:
            print("  -", p)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
