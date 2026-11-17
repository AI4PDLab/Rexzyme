#!/usr/bin/env python3
"""Plot sequence identity to the training set against best TM-score.

Uses the all-designs table from ``plot_tmscore_vs_plddt.py``. Keeps designs
with mean pLDDT >= 70, then one point per held-out reaction. Sequence
identity comes from ``designs_vs_train.best.tsv``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    METRIC_LABELS,
    TM_METRIC_CHOICES,
    apply_style,
    configure_tm_axes,
    correlations_and_fit,
    design_stem,
    plot_linear_fit,
    reaction_number,
    read_best_designs_after_plddt_filter,
    required_columns,
    save_figure,
    scatter_heldout,
)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def parse_selected_design_index(design_name: str) -> str | None:
    name = design_stem(design_name)
    match = re.search(r"(?:^|_)(\d+)_ppl", name)
    return match.group(1) if match else None


def parse_mmseqs_design(design_name: str) -> tuple[str, str] | None:
    match = re.search(r"output_file[_-](\d+)_(\d+)_ppl", design_name)
    if match is None:
        return None
    return match.group(1), match.group(2)


def read_sequence_identities(mmseqs_path: Path) -> dict[tuple[str, str], float]:
    sequence_identity = {}
    with mmseqs_path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        columns = required_columns(header, {"design", "pident"}, mmseqs_path)
        max_index = max(columns["design"], columns["pident"])
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= max_index:
                continue
            parsed = parse_mmseqs_design(fields[columns["design"]])
            if parsed is None:
                continue
            try:
                pident = float(fields[columns["pident"]])
            except ValueError:
                continue
            if not np.isfinite(pident):
                continue
            sequence_identity[parsed] = pident
    print(f"Designs with a parsed MMseqs sequence-identity value: {len(sequence_identity):,}")
    return sequence_identity


def main() -> None:
    apply_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plddt-tsv", required=True, type=Path)
    parser.add_argument(
        "--mmseqs",
        required=True,
        type=Path,
        help="designs_vs_train.best.tsv",
    )
    parser.add_argument("--out-prefix", "--output", dest="out_prefix", required=True, type=Path,
                        help="Output prefix for the TSV and figure files")
    parser.add_argument("--tm-metric", default="ttmscore", choices=TM_METRIC_CHOICES)
    parser.add_argument("--min-plddt", type=float, default=70.0)
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Plot designs with no MMseqs hit at sequence identity 0",
    )
    parser.add_argument("--point-size", type=float, default=85)
    parser.add_argument("--point-alpha", type=float, default=0.62)
    parser.add_argument("--edge-width", type=float, default=0.7)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    selected = read_best_designs_after_plddt_filter(
        args.plddt_tsv, args.tm_metric, args.min_plddt
    )
    sequence_identity = read_sequence_identities(args.mmseqs)

    xs, ys, rows = [], [], []
    missing_hits = invalid_names = 0
    for reaction in sorted(selected):
        values = selected[reaction]
        number = reaction_number(reaction)
        design_index = parse_selected_design_index(values["design"])
        if number is None or design_index is None:
            invalid_names += 1
            continue
        pident = sequence_identity.get((str(number), design_index))
        if pident is None:
            missing_hits += 1
            if not args.include_missing:
                continue
            pident = 0.0
        xs.append(pident)
        ys.append(values["tm_score"])
        rows.append(
            (reaction, values["design"], values["plddt"], pident, values["tm_score"])
        )

    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if len(xs) < 2:
        raise SystemExit("Fewer than two matched reactions after the pLDDT filter.")
    print(f"Reactions plotted: {len(xs):,}")
    print(f"Selected designs without an MMseqs hit: {missing_hits:,}")
    print(f"Selected designs with unrecognized names: {invalid_names:,}")

    fit = correlations_and_fit(xs, ys)
    tsv_path = Path(f"{args.out_prefix}.tsv")
    with tsv_path.open("w", encoding="utf-8") as handle:
        handle.write(
            f"reaction\tbest_design\tmean_plddt\tseq_identity_pct\tbest_{args.tm_metric}\n"
        )
        for reaction, design, plddt, pident, tm_score in rows:
            handle.write(
                f"{reaction}\t{design}\t{plddt:.3f}\t{pident:.3f}\t{tm_score:.4f}\n"
            )
    print(f"Saved data: {tsv_path}")

    fig, ax = plt.subplots(figsize=(7.6, 6.3), constrained_layout=True)
    scatter_heldout(ax, xs, ys, args.point_size, args.point_alpha, args.edge_width)
    plot_linear_fit(ax, xs, fit)
    ax.set_title("Sequence similarity versus structural similarity", fontweight="semibold", pad=14)
    ax.text(
        0.5,
        1.01,
        f"Best design per reaction (mean pLDDT ≥ {args.min_plddt:g})",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="0.35",
    )
    ax.set_xlabel("Sequence identity to closest training sequence (%)", labelpad=8)
    ax.set_ylabel(
        "Best TM-score to ground truth\n"
        f"({METRIC_LABELS[args.tm_metric]})",
        labelpad=9,
    )
    configure_tm_axes(ax, x_max=100, x_min=-1)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    save_figure(fig, args.out_prefix, args.dpi)


if __name__ == "__main__":
    main()
