#!/usr/bin/env python3
"""
Plot embedding distance against the best design TM-score.

Processing order:
    1. Read all designs from the pLDDT / TM-score TSV.
    2. Keep only designs with mean pLDDT >= the chosen threshold.
    3. For each reaction, select the retained design with the highest TM-score.
    4. Match each reaction to its embedding distance.
    5. Plot one point per reaction.
    6. Plot a linear least-squares fit.

Important:
    The pLDDT TSV must contain all designs. It should have been generated
    without the --best-per-reaction option.

Expected embedding-distance input columns:
    file_name
    min_emb_distance
    smiles

Outputs:
    <out-prefix>.tsv
    <out-prefix>.png
    <out-prefix>.pdf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    LINEAR_FIT_COLOR,
    METRIC_LABELS,
    TM_METRIC_CHOICES,
    VAL_DIF_COLOR,
    VAL_DIF_DARK,
    apply_style,
    reaction_number,
    read_best_designs_after_plddt_filter,
    required_columns,
    save_figure,
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (
    AutoMinorLocator,
    FormatStrFormatter,
    MaxNLocator,
    MultipleLocator,
)
from scipy.stats import pearsonr, spearmanr


def read_embedding_distances(tsv_path: Path) -> dict[int, float]:
    """
    Read embedding_distance_by_file.tsv.

    Expected columns:
        file_name
        min_emb_distance
        smiles

    Returns:
        reaction number -> minimum embedding distance
    """
    embedding_by_number = {}
    with tsv_path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        columns = required_columns(header, {"file_name", "min_emb_distance"}, tsv_path)
        file_index = columns["file_name"]
        dist_index = columns["min_emb_distance"]
        maximum_index = max(file_index, dist_index)
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= maximum_index:
                continue
            match = re.search(r"output_file[_-](\d+)\.fasta", fields[file_index])
            if match is None:
                continue
            try:
                emb_distance = float(fields[dist_index])
            except ValueError:
                continue
            if not np.isfinite(emb_distance):
                continue
            embedding_by_number[int(match.group(1))] = emb_distance
    return embedding_by_number


def merge_reaction_data(selected, embedding_by_number):
    """Match each selected design to the embedding distance of its reaction."""
    xs = []
    ys = []
    rows = []
    missing_embedding = 0
    # Sort by parsed reaction index so mixed naming styles
    # (output_file_12 vs output_file-12) stay in numeric order.
    def reaction_sort_key(reaction: str) -> tuple[int, int | str]:
        number = reaction_number(reaction)
        if number is None:
            return (1, reaction)
        return (0, number)

    for reaction in sorted(selected, key=reaction_sort_key):
        number = reaction_number(reaction)
        if number is None:
            continue
        if number not in embedding_by_number:
            missing_embedding += 1
            continue
        values = selected[reaction]
        emb_distance = embedding_by_number[number]
        tm_score = values["tm_score"]
        design = values["design"]
        plddt = values["plddt"]
        xs.append(emb_distance)
        ys.append(tm_score)
        rows.append((reaction, design, plddt, emb_distance, tm_score))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), rows, missing_embedding


def save_data(rows, output_path: Path, tm_metric: str) -> None:
    """Save the exact data displayed in the plot."""
    with output_path.open("w", encoding="utf-8") as output:
        output.write(
            "reaction\tbest_design\tmean_plddt\tmin_emb_distance\t"
            f"best_{tm_metric}\n"
        )
        for reaction, design, plddt, emb_distance, tm_score in rows:
            output.write(
                f"{reaction}\t{design}\t{plddt:.3f}\t{emb_distance:.6f}\t{tm_score:.4f}\n"
            )


def configure_axes(ax, xs) -> None:
    """Apply publication-style axis formatting."""
    x_upper = np.ceil((xs.max() + 0.02) * 10) / 10
    x_upper = max(x_upper, 0.1)
    ax.set_xlim(-0.01, x_upper)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9, min_n_ticks=5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(axis="both", which="major", length=5, width=0.8)
    ax.tick_params(axis="both", which="minor", length=3, width=0.6)
    ax.set_axisbelow(True)
    ax.grid(which="major", linewidth=0.7, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    apply_style()
    parser = argparse.ArgumentParser(
        description=(
            "Filter designs by pLDDT, select the best design per reaction, "
            "and plot TM-score against embedding distance."
        )
    )
    parser.add_argument(
        "--plddt-tsv",
        required=True,
        type=Path,
        help="Per-design table containing reaction, design, mean_plddt, and best TM-score.",
    )
    parser.add_argument(
        "--embedding-tsv",
        required=True,
        type=Path,
        help="TSV containing file_name and min_emb_distance (embedding_distance_by_file.tsv).",
    )
    parser.add_argument("--tm-metric", default="ttmscore", choices=TM_METRIC_CHOICES)
    parser.add_argument(
        "--min-plddt",
        type=float,
        default=70.0,
        help="Minimum mean pLDDT. Designs at or above this value are retained.",
    )
    parser.add_argument("--out-prefix", "--output", dest="out_prefix", required=True, type=Path,
                        help="Output prefix for the TSV and figure files")
    parser.add_argument("--point-size", type=float, default=85)
    parser.add_argument("--point-alpha", type=float, default=0.62)
    parser.add_argument("--edge-width", type=float, default=0.7)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    output_prefix = Path(args.out_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    selected = read_best_designs_after_plddt_filter(
        plddt_tsv=args.plddt_tsv,
        tm_metric=args.tm_metric,
        minimum_plddt=args.min_plddt,
    )
    embedding_by_number = read_embedding_distances(args.embedding_tsv)
    xs, ys, rows, missing_embedding = merge_reaction_data(
        selected, embedding_by_number
    )

    if len(xs) < 2:
        raise RuntimeError(
            "Fewer than two matched reactions were found after applying the pLDDT filter."
        )

    print(f"Reactions plotted: {len(xs):,}")
    print(f"Selected reactions without embedding distance: {missing_embedding:,}")

    pearson_r, pearson_p = pearsonr(xs, ys)
    spearman_rho, spearman_p = spearmanr(xs, ys)
    print(f"Pearson r = {pearson_r:.3f}, p = {pearson_p:.3e}")
    print(f"Spearman rho = {spearman_rho:.3f}, p = {spearman_p:.3e}")

    slope, intercept = np.polyfit(xs, ys, deg=1)
    r_squared = pearson_r ** 2
    fit_x = np.linspace(xs.min(), xs.max(), 300)
    fit_y = slope * fit_x + intercept
    print(f"Linear fit: y = {slope:.4f}x + {intercept:.4f}")
    print(f"Linear fit R^2 = {r_squared:.4f}")

    tsv_path = Path(f"{output_prefix}.tsv")
    save_data(rows, tsv_path, args.tm_metric)
    print(f"Saved data: {tsv_path}")

    fig, ax = plt.subplots(figsize=(7.6, 6.3), constrained_layout=True)
    ax.scatter(
        xs,
        ys,
        s=args.point_size,
        facecolors=VAL_DIF_COLOR,
        edgecolors=VAL_DIF_DARK,
        linewidths=args.edge_width,
        alpha=args.point_alpha,
        rasterized=True,
        zorder=2,
    )
    ax.plot(
        fit_x,
        fit_y,
        color=LINEAR_FIT_COLOR,
        linestyle="--",
        linewidth=2.2,
        alpha=0.90,
        zorder=3,
        label="Linear fit",
    )
    ax.set_title(
        "Reaction novelty versus structural similarity",
        fontweight="semibold",
        pad=14,
    )
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
    ax.set_xlabel("Reaction embedding distance to training set", labelpad=8)
    ax.set_ylabel(
        "Best TM-score to ground truth\n"
        f"({METRIC_LABELS[args.tm_metric]})",
        labelpad=9,
    )
    configure_axes(ax=ax, xs=xs)
    save_figure(fig, output_prefix, args.dpi)


if __name__ == "__main__":
    main()
