#!/usr/bin/env python3
"""Re-derive filtered TM-score tables from unfiltered Foldseek output.

Keep a design <-> ground-truth pair only if:

  tcov  >= MIN_TCOV   (fraction of the ground-truth chain covered)
  tlen  >= MIN_TLEN   (ground-truth chain is not a tiny peptide)
  qcov  >= MIN_QCOV   (fraction of the design covered)

Rewrites:

  all_tmscores.tsv
  output_file_<N>/tmscores.tsv   (from tmscores_all.tsv, including 1255)
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

MIN_TCOV = 0.5
MIN_TLEN = 50
MIN_QCOV = 0.5


def keep(row: list[str], col: dict[str, int], min_tcov: float, min_tlen: float, min_qcov: float) -> bool:
    try:
        return (
            float(row[col["tcov"]]) >= min_tcov
            and float(row[col["tlen"]]) >= min_tlen
            and float(row[col["qcov"]]) >= min_qcov
        )
    except (ValueError, KeyError, IndexError):
        return False


def filter_file(
    in_path: Path,
    out_path: Path,
    min_tcov: float,
    min_tlen: float,
    min_qcov: float,
) -> tuple[int, int]:
    with in_path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        kept = 0
        total = 0
        with out_path.open("w", encoding="utf-8") as out:
            out.write("\t".join(header) + "\n")
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < len(header):
                    continue
                total += 1
                if keep(parts, col, min_tcov, min_tlen, min_qcov):
                    out.write(line if line.endswith("\n") else line + "\n")
                    kept += 1
    return kept, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Directory containing all_tmscores_unfiltered.tsv and output_file_* folders",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Unfiltered aggregate TSV (default: <root>/all_tmscores_unfiltered.tsv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Filtered aggregate TSV (default: <root>/all_tmscores.tsv)",
    )
    parser.add_argument("--min-tcov", type=float, default=MIN_TCOV)
    parser.add_argument("--min-tlen", type=float, default=MIN_TLEN)
    parser.add_argument("--min-qcov", type=float, default=MIN_QCOV)
    args = parser.parse_args()

    unfiltered = args.input or (args.root / "all_tmscores_unfiltered.tsv")
    filtered = args.output or (args.root / "all_tmscores.tsv")
    filtered.parent.mkdir(parents=True, exist_ok=True)
    if not unfiltered.is_file():
        raise SystemExit(f"Unfiltered aggregate not found: {unfiltered}")

    kept, total = filter_file(
        unfiltered,
        filtered,
        args.min_tcov,
        args.min_tlen,
        args.min_qcov,
    )
    print(
        f"aggregate: kept {kept} / {total} pairs "
        f"(removed {total - kept}); wrote {filtered}"
    )

    n_folders = 0
    for all_file in sorted(glob.glob(str(args.root / "output_file_*" / "tmscores_all.tsv"))):
        all_path = Path(all_file)
        filter_file(
            all_path,
            all_path.parent / "tmscores.tsv",
            args.min_tcov,
            args.min_tlen,
            args.min_qcov,
        )
        n_folders += 1
    print(f"rewrote per-reaction tmscores.tsv in {n_folders} folders")


if __name__ == "__main__":
    main()
