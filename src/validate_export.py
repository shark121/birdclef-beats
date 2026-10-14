"""Phase 4 gate: the plan's section 3.5 checklist, as an executable test.

Run this before any export is loaded or shared. Exits non-zero on any failure
so it can sit in front of an upload step.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a projector export.")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--dir", type=Path, default=None)
    ap.add_argument("--expect-order-from", type=Path, default=None,
                    help="clip_order csv the export was built from (identity check)")
    args = ap.parse_args()

    out_dir = args.dir or (config.EXPORT_DIR / args.tag)
    vectors_tsv = out_dir / "vectors.tsv"
    metadata_tsv = out_dir / "metadata.tsv"

    if not vectors_tsv.exists() or not metadata_tsv.exists():
        print(f"FAIL: missing vectors.tsv / metadata.tsv in {out_dir}")
        return 1

    results = []

    def check(ok, label, detail=""):
        results.append((bool(ok), label + (f" -- {detail}" if detail else "")))

    # ---- vectors.tsv: uniform width, finite parseable values ---------------
    widths = set()
    n_vec = 0
    bad_values = 0
    with open(vectors_tsv, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            widths.add(len(fields))
            n_vec += 1
            for field in fields:
                try:
                    value = float(field)
                except ValueError:
                    bad_values += 1
                    continue
                if math.isnan(value) or math.isinf(value):
                    bad_values += 1

    check(len(widths) == 1, "vectors.tsv has one consistent column count",
          f"widths seen: {sorted(widths)}")
    with open(vectors_tsv, "rb") as fh:
        check(b"\r" not in fh.read(), "vectors.tsv has no stray carriage returns")
    check(bad_values == 0, "vectors.tsv has no NaN/Inf/unparseable values",
          f"{bad_values} bad values")

    # ---- metadata.tsv: header, matching widths, clean cells ----------------
    with open(metadata_tsv, encoding="utf-8", newline="") as fh:
        raw = fh.read()
    check("\r" not in raw, "metadata.tsv has no stray carriage returns")

    lines = [ln for ln in raw.split("\n") if ln.strip() != ""]
    header = lines[0].split("\t")
    data_rows = lines[1:]
    ragged = [i for i, ln in enumerate(data_rows, start=2)
              if len(ln.split("\t")) != len(header)]

    check(not ragged, "every metadata row matches the header width",
          f"{len(ragged)} ragged rows, first at line {ragged[0] if ragged else 0}")
    check(n_vec == len(data_rows),
          "row counts match (vectors.tsv == metadata.tsv minus header)",
          f"{n_vec} vs {len(data_rows)}")
    check("clip_id" in header, "metadata.tsv exposes clip_id for spot-checks")

    # ---- positional identity against the embedding build order -------------
    if args.expect_order_from and args.expect_order_from.exists():
        import pandas as pd
        expected = pd.read_csv(args.expect_order_from)["clip_id"].astype(str).tolist()
        col = header.index("clip_id")
        actual = [row.split("\t")[col] for row in data_rows]
        if len(expected) != len(actual):
            check(False, "export length equals embedding order length",
                  f"{len(actual)} vs {len(expected)}")
        else:
            diverge = None
            for i, (a, b) in enumerate(zip(expected, actual)):
                if a != b:
                    diverge = i
                    break
            check(diverge is None,
                  "row i of the export is row i of the embedding order",
                  f"first divergence at row {diverge}")

    print(f"validating {out_dir}\n")
    for ok, label in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    n_dim = next(iter(widths)) if len(widths) == 1 else -1
    print(f"\n  points={n_vec}  dimensions={n_dim}  metadata columns={len(header)}")

    failed = [label for ok, label in results if not ok]
    if failed:
        print(f"\n{len(failed)} CHECK(S) FAILED -- do not load this export")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
