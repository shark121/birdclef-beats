"""Phase 2 acceptance check: are the clips actually bird song?

The plan asks for a listen-through of 20 random clips. Standing in for ears,
this checks the things that would betray silence, hum or a dead file:

  * the audio contract holds exactly (5.0 s, 16 kHz, mono)
  * the clip is not near-silent (RMS floor)
  * most spectral energy sits in the 1.5-9 kHz band where passerine song lives,
    rather than in sub-500 Hz rumble (wind, traffic, handling noise)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config

BIRD_LO, BIRD_HI = 1500.0, 9000.0
RUMBLE_HI = 500.0


def band_fractions(audio: np.ndarray, sr: int) -> tuple[float, float, float]:
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) ** 2
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    total = spec.sum() + 1e-12
    bird = spec[(freqs >= BIRD_LO) & (freqs <= BIRD_HI)].sum() / total
    rumble = spec[freqs < RUMBLE_HI].sum() / total
    centroid = float((freqs * spec).sum() / total)
    return float(bird), float(rumble), centroid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    args = ap.parse_args()

    clips_csv = config.INTERIM_DIR / "clips.csv"
    if not clips_csv.exists():
        print(f"missing {clips_csv}")
        return 1
    clips = pd.read_csv(clips_csv)
    sample = clips.sample(n=min(args.n, len(clips)), random_state=args.seed)

    print(f"spot-checking {len(sample)} random clips of {len(clips):,}\n")
    print(f"{'clip_id':<44} {'sp':<7} {'sec':>5} {'sr':>6} {'rms':>7} "
          f"{'bird%':>6} {'rumble%':>8} {'cent Hz':>8}")
    print("-" * 100)

    contract_ok = True
    bird_fracs, flagged = [], []
    for _, row in sample.iterrows():
        path = config.CLIPS_DIR / row["clip_path"]
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        dur = len(audio) / sr
        rms = float(np.sqrt(np.mean(audio ** 2)))
        bird, rumble, centroid = band_fractions(audio, sr)
        bird_fracs.append(bird)

        ok = (abs(dur - config.CLIP_SECONDS) < 1e-3
              and sr == config.SAMPLE_RATE_HZ
              and rms >= config.MIN_CLIP_RMS)
        contract_ok &= ok
        mark = "" if bird >= 0.25 else "  <- low band energy"
        if bird < 0.25:
            flagged.append(row["clip_id"])
        print(f"{row['clip_id']:<44} {row['species_code']:<7} {dur:>5.2f} {sr:>6} "
              f"{rms:>7.4f} {bird:>5.0%} {rumble:>7.0%} {centroid:>8.0f}{mark}")

    print("\n" + "-" * 100)
    print(f"audio contract (5.0s / 16 kHz / mono / above RMS floor): "
          f"{'ALL PASS' if contract_ok else 'FAILURES PRESENT'}")
    print(f"median energy in {BIRD_LO/1000:.1f}-{BIRD_HI/1000:.0f} kHz bird band: "
          f"{np.median(bird_fracs):.0%}")
    print(f"clips dominated by sub-{RUMBLE_HI:.0f} Hz rumble instead: {len(flagged)}"
          + (f"  ({', '.join(flagged[:4])})" if flagged else ""))
    return 0 if contract_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
