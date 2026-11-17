#!/usr/bin/env python3
"""Plot mean design pLDDT against best TM-score to ground truth.

One point per design (or, with ``--best-per-reaction``, one point per
held-out reaction). pLDDT is the mean CA B-factor of the folded PDB.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    DENSITY_CMAPS,
    METRIC_LABELS,
    PAPER_COLORS,
    TM_METRIC_CHOICES,
    apply_style,
    configure_tm_axes,
    correlations_and_fit,
    mean_plddt,
    plot_linear_fit,
    read_best_tmscores,
    resolve_design_pdb,
    retain_best_design_per_reaction,
    save_figure,
)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MultipleLocator


def collect_points(best, designs_root: Path):
    cache = {}
    xs, ys, rows = [], [], []
    missing = 0
    for (reaction, design), tm_score in best.items():
        pdb_path = resolve_design_pdb(designs_root, reaction, design)
        if pdb_path is None:
            missing += 1
            continue
        if pdb_path not in cache:
            cache[pdb_path] = mean_plddt(pdb_path)
        plddt = cache[pdb_path]
        if plddt is None or not np.isfinite(plddt):
            missing += 1
            continue
        xs.append(plddt)
        ys.append(tm_score)
        rows.append((reaction, design, plddt, tm_score))
    return np.asarray(xs, float), np.asarray(ys, float), rows, missing


def main() -> None:
    apply_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", required=True, type=Path, help="all_tmscores.tsv")
    parser.add_argument(
        "--designs",
        required=True,
        type=Path,
        help="Root of folded designs (output_file_<N>/pdbs/*.pdb)",
    )
    parser.add_argument("--out-prefix", "--output", dest="out_prefix", required=True, type=Path,
                        help="Output prefix for the TSV and figure files")
    parser.add_argument("--tm-metric", default="ttmscore", choices=TM_METRIC_CHOICES)
    parser.add_argument(
        "--best-per-reaction",
        action="store_true",
        help="Keep only the highest-TM-score design per reaction",
    )
    parser.add_argument("--palette", default="val_dif", choices=list(PAPER_COLORS))
    parser.add_argument("--gridsize", type=int, default=65)
    parser.add_argument("--hexbin-min-count", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    best = read_best_tmscores(args.master, args.tm_metric)
    if args.best_per_reaction:
        best = retain_best_design_per_reaction(best)

    xs, ys, rows, missing = collect_points(best, args.designs)
    if len(xs) < 2:
        raise SystemExit("Fewer than two valid designs were found.")
    print(f"Designs plotted: {len(xs):,}")
    print(f"Missing or invalid PDB files: {missing:,}")

    fit = correlations_and_fit(xs, ys)

    tsv_path = Path(f"{args.out_prefix}.tsv")
    with tsv_path.open("w", encoding="utf-8") as handle:
        handle.write(f"reaction\tdesign\tmean_plddt\tbest_{args.tm_metric}\n")
        for reaction, design, plddt, tm_score in rows:
            handle.write(f"{reaction}\t{design}\t{plddt:.3f}\t{tm_score:.4f}\n")
    print(f"Saved data: {tsv_path}")

    fig, ax = plt.subplots(figsize=(7.6, 6.4), constrained_layout=True)
    if args.best_per_reaction:
        ax.scatter(
            xs,
            ys,
            s=32,
            color=PAPER_COLORS[args.palette],
            alpha=0.70,
            edgecolors="white",
            linewidths=0.4,
            rasterized=True,
        )
        scope_label = "Best design per reaction"
    else:
        hexbin = ax.hexbin(
            xs,
            ys,
            gridsize=args.gridsize,
            extent=(0, 100, 0, 1),
            mincnt=args.hexbin_min_count,
            cmap=DENSITY_CMAPS[args.palette],
            norm=LogNorm(vmin=1),
            linewidths=0,
            rasterized=True,
        )
        colorbar = fig.colorbar(hexbin, ax=ax, pad=0.025, fraction=0.055, aspect=28)
        colorbar.set_label("Number of designs per hexagon", labelpad=10)
        colorbar.locator = LogLocator(base=10)
        colorbar.formatter = LogFormatterMathtext(base=10)
        colorbar.update_ticks()
        colorbar.outline.set_linewidth(0.8)
        scope_label = "All designs"

    plot_linear_fit(ax, xs, fit)
    ax.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="0.75",
        framealpha=0.92,
        fontsize=10,
    )
    ax.set_title("Design confidence versus structural similarity", fontweight="semibold", pad=12)
    ax.text(
        0.5,
        1.01,
        scope_label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="0.35",
    )
    ax.set_xlabel("Mean design pLDDT", labelpad=8)
    ax.set_ylabel(
        "Best TM-score to ground truth\n"
        f"({METRIC_LABELS[args.tm_metric]})",
        labelpad=9,
    )
    configure_tm_axes(ax, x_max=100, x_min=0)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    save_figure(fig, args.out_prefix, args.dpi)


if __name__ == "__main__":
    main()
