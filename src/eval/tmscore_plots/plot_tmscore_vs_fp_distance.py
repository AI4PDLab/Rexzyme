#!/usr/bin/env python3
"""Plot fingerprint distance against the best confident design TM-score.

Uses the all-designs table from ``plot_tmscore_vs_plddt.py`` (do not pass
``--best-per-reaction`` when creating that table). Keeps designs with
mean pLDDT >= 70, then one point per held-out reaction.
"""

from __future__ import annotations

import argparse
import json
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
    plot_linear_fit,
    reaction_number,
    read_best_designs_after_plddt_filter,
    save_figure,
    scatter_heldout,
)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


def read_fingerprint_distances(json_path: Path) -> dict[int, float]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    by_number = {}
    for entry in data:
        match = re.search(r"output_file[_-](\d+)\.fasta", str(entry.get("fasta_path", "")))
        if match is None:
            continue
        try:
            minimum_distance = float(entry["min_distance"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(minimum_distance):
            continue
        by_number[int(match.group(1))] = minimum_distance
    return by_number


def main() -> None:
    apply_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plddt-tsv",
        required=True,
        type=Path,
        help="All-designs TSV from plot_tmscore_vs_plddt.py",
    )
    parser.add_argument(
        "--fp",
        required=True,
        type=Path,
        help="fingerprint_distances.json",
    )
    parser.add_argument("--out-prefix", "--output", dest="out_prefix", required=True, type=Path,
                        help="Output prefix for the TSV and figure files")
    parser.add_argument("--tm-metric", default="ttmscore", choices=TM_METRIC_CHOICES)
    parser.add_argument("--min-plddt", type=float, default=70.0)
    parser.add_argument("--point-size", type=float, default=85)
    parser.add_argument("--point-alpha", type=float, default=0.62)
    parser.add_argument("--edge-width", type=float, default=0.7)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    selected = read_best_designs_after_plddt_filter(
        args.plddt_tsv, args.tm_metric, args.min_plddt
    )
    fp_by_number = read_fingerprint_distances(args.fp)

    xs, ys, rows = [], [], []
    missing = 0
    for reaction in sorted(selected):
        number = reaction_number(reaction)
        if number is None or number not in fp_by_number:
            missing += 1
            continue
        values = selected[reaction]
        xs.append(fp_by_number[number])
        ys.append(values["tm_score"])
        rows.append(
            (
                reaction,
                values["design"],
                values["plddt"],
                fp_by_number[number],
                values["tm_score"],
            )
        )
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if len(xs) < 2:
        raise SystemExit("Fewer than two matched reactions after the pLDDT filter.")
    print(f"Reactions plotted: {len(xs):,}")
    print(f"Selected reactions without fingerprint distance: {missing:,}")

    fit = correlations_and_fit(xs, ys)
    tsv_path = Path(f"{args.out_prefix}.tsv")
    with tsv_path.open("w", encoding="utf-8") as handle:
        handle.write(
            f"reaction\tbest_design\tmean_plddt\tmin_distance\tbest_{args.tm_metric}\n"
        )
        for reaction, design, plddt, dist, tm_score in rows:
            handle.write(
                f"{reaction}\t{design}\t{plddt:.3f}\t{dist:.6f}\t{tm_score:.4f}\n"
            )
    print(f"Saved data: {tsv_path}")

    fig, ax = plt.subplots(figsize=(7.6, 6.3), constrained_layout=True)
    scatter_heldout(ax, xs, ys, args.point_size, args.point_alpha, args.edge_width)
    plot_linear_fit(ax, xs, fit)
    ax.set_title("Reaction novelty versus structural similarity", fontweight="semibold", pad=14)
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
    ax.set_xlabel("Reaction fingerprint distance to training set", labelpad=8)
    ax.set_ylabel(
        "Best TM-score to ground truth\n"
        f"({METRIC_LABELS[args.tm_metric]})",
        labelpad=9,
    )
    x_upper = max(np.ceil((xs.max() + 0.02) * 10) / 10, 0.1)
    configure_tm_axes(ax, x_max=x_upper, x_min=-0.01)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9, min_n_ticks=5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    save_figure(fig, args.out_prefix, args.dpi)


if __name__ == "__main__":
    main()
