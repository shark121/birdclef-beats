"""Phase 5: quantify what the projection shows, so "which species cluster"
is answered with numbers rather than only by eye.

The one methodological trap this guards against: clips cut from the SAME
recording are near-duplicates of each other. Let them count as neighbours and
species purity looks superb for reasons that have nothing to do with the model
generalising. Every neighbour test here excludes same-recording neighbours.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config


def knn_purity(vectors: np.ndarray, labels: np.ndarray, groups: np.ndarray,
               k: int = 10, block: int = 512) -> tuple[float, dict[str, float]]:
    """Fraction of a point's k nearest neighbours sharing its label, where
    neighbours from the same recording (`groups`) are excluded entirely."""
    normed = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)
    n = len(normed)
    per_point = np.zeros(n, dtype=np.float64)
    # Integer codes so the per-block comparisons are vectorised, not per point:
    # the per-point loop this replaced was fine at 12k clips and hours at 150k.
    _, gid = np.unique(groups, return_inverse=True)
    _, lid = np.unique(labels, return_inverse=True)

    for start in range(0, n, block):
        stop = min(start + block, n)
        sims = normed[start:stop] @ normed.T                  # cosine similarity
        sims[gid[start:stop, None] == gid[None, :]] = -np.inf  # drop same recording
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        usable = np.isfinite(np.take_along_axis(sims, top, axis=1))
        same = (lid[top] == lid[start:stop, None]) & usable
        n_usable = usable.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_point[start:stop] = np.where(n_usable > 0, same.sum(axis=1) / n_usable, np.nan)

    valid = ~np.isnan(per_point)
    overall = float(per_point[valid].mean())
    by_label: dict[str, float] = {}
    for lab in np.unique(labels):
        mask = valid & (labels == lab)
        if mask.any():
            by_label[str(lab)] = float(per_point[mask].mean())
    return overall, by_label


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyse embedding cluster quality.")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    emb_path = config.EMB_DIR / f"embeddings_{args.tag}.npy"
    order_path = config.EMB_DIR / f"clip_order_{args.tag}.csv"
    if not emb_path.exists():
        print(f"missing {emb_path}")
        return 1

    vectors = np.load(emb_path).astype(np.float32)
    # BirdCLEF 2026 mixes eBird codes with numeric iNat taxon ids; read labels as
    # text or pandas may make some of them ints and break label equality.
    clips = pd.read_csv(order_path, dtype={"species_code": str}).fillna("")
    species = clips["species_code"].to_numpy()
    sources = clips["dataset_source"].to_numpy()
    recordings = clips["recording_id"].to_numpy()

    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say(f"# Embedding analysis -- tag `{args.tag}`")
    say()
    say(f"- points: **{len(vectors):,}** clips from **{len(np.unique(recordings)):,}** recordings")
    say(f"- dimensions: **{vectors.shape[1]}**")
    say(f"- species: **{len(np.unique(species))}**")
    say(f"- sources: {', '.join(sorted(set(sources)))}")
    say()

    # ---- PCA spectrum: how much structure sits in the first few axes -------
    centred = vectors - vectors.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centred, compute_uv=False)
    var = singular ** 2
    var = var / var.sum()
    say(f"- PCA variance explained: PC1-2 **{var[:2].sum():.1%}**, "
        f"PC1-10 **{var[:10].sum():.1%}**, PC1-50 **{var[:50].sum():.1%}**")
    say()

    # ---- species separability ---------------------------------------------
    chance = float((pd.Series(species).value_counts(normalize=True) ** 2).sum())
    overall, by_species = knn_purity(vectors, species, recordings, k=args.k)
    say(f"## Species separability (k={args.k} NN, same-recording neighbours excluded)")
    say()
    say(f"- **{overall:.1%}** of neighbours share the query's species "
        f"(chance = {chance:.1%}, so **{overall / chance:.1f}x** chance)")
    say()

    ranked = sorted(by_species.items(), key=lambda kv: kv[1], reverse=True)
    name_of = {s["species_code"]: s["common_name"] for s in config.load_species()}
    say("| rank | species | common name | kNN purity |")
    say("|---|---|---|---|")
    for i, (code, score) in enumerate(ranked, 1):
        say(f"| {i} | `{code}` | {name_of.get(code, '?')} | {score:.1%} |")
    say()
    say(f"**Tightest:** {', '.join(f'{c} ({v:.0%})' for c, v in ranked[:3])}")
    say(f"**Most overlapped:** {', '.join(f'{c} ({v:.0%})' for c, v in ranked[-3:])}")
    say()

    # ---- by taxon class (BirdCLEF 2026: birds vs frogs vs insects ...) -----
    if "taxon_class" in clips.columns and clips["taxon_class"].astype(bool).any():
        class_of = dict(zip(clips["species_code"], clips["taxon_class"]))
        say("## Species purity by taxon class")
        say()
        say("| class | species | clips | mean species purity |")
        say("|---|---|---|---|")
        counts = clips["taxon_class"].value_counts()
        for cls in counts.index:
            scores = [v for c, v in by_species.items() if class_of.get(c) == cls]
            say(f"| {cls} | {len(scores)} | {counts[cls]:,} | {np.mean(scores):.1%} |")
        say()

    # ---- dataset-source batch effect --------------------------------------
    say("## Dataset-source batch effect")
    say()
    if len(set(sources)) < 2:
        say(f"Only one source present (`{sources[0]}`), so this test is not yet "
            "meaningful. Re-run once Xeno-canto, BirdCLEF or field audio is added: "
            "if source purity lands far above chance while species purity does not, "
            "the projection is partly organised by *where the audio came from* "
            "rather than by *which bird is singing*.")
    else:
        src_chance = float((pd.Series(sources).value_counts(normalize=True) ** 2).sum())
        src_overall, by_source = knn_purity(vectors, sources, recordings, k=args.k)
        say(f"- **{src_overall:.1%}** of neighbours share the query's source "
            f"(chance = {src_chance:.1%}, **{src_overall / src_chance:.1f}x**)")
        say()
        for code, score in sorted(by_source.items(), key=lambda kv: -kv[1]):
            say(f"  - `{code}`: {score:.1%}")
        say()
        verdict = ("a visible batch effect -- treat cross-source clusters with care"
                   if src_overall / src_chance > 1.5 else
                   "no strong batch effect at this k")
        say(f"Reading: {verdict}.")
    say()

    out = args.out or (config.EXPORT_DIR / args.tag / "analysis.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
