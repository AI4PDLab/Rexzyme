#!/usr/bin/env python3
"""Plot MMseqs sequence identity versus alignment length.

Compare generated sequences against the training set (see
``mmseq_file_preparation.py`` and ``mmseq_interpretation.py`` in this
folder). Expected input is an MMseqs tabular file with at least:

    query  target  sequence_identity  alignment_length
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb


def read_mmseqs_file(
    input_file: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read sequence identity and alignment length from an MMseqs output.

    Expected columns:
        1: query
        2: target
        3: sequence identity
        4: alignment length

    Returns:
        alignment_lengths:
            Array containing alignment lengths.

        sequence_identities:
            Array containing sequence identities on a 0-1 scale.
    """
    sequence_identities: list[float] = []
    alignment_lengths: list[int] = []

    malformed_count = 0

    with input_file.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            fields = line.split()

            if len(fields) < 4:
                malformed_count += 1
                print(
                    f"Warning: skipped malformed line "
                    f"{line_number}: fewer than four columns"
                )
                continue

            try:
                sequence_identity = float(fields[2])
                alignment_length = int(float(fields[3]))

            except ValueError:
                malformed_count += 1
                print(
                    f"Warning: invalid numerical values "
                    f"on line {line_number}"
                )
                continue

            # Convert percentage values such as 67.7 to 0.677.
            if sequence_identity > 1.5:
                sequence_identity /= 100.0

            if not 0 <= sequence_identity <= 1:
                malformed_count += 1
                print(
                    f"Warning: sequence identity outside 0-1 "
                    f"on line {line_number}: "
                    f"{sequence_identity}"
                )
                continue

            if alignment_length <= 0:
                malformed_count += 1
                print(
                    f"Warning: non-positive alignment length "
                    f"on line {line_number}: "
                    f"{alignment_length}"
                )
                continue

            sequence_identities.append(sequence_identity)
            alignment_lengths.append(alignment_length)

    if not sequence_identities:
        raise ValueError(
            f"No valid alignments were found in {input_file}"
        )

    if malformed_count:
        print(
            f"Skipped malformed alignments: "
            f"{malformed_count:,}"
        )

    return (
        np.array(alignment_lengths, dtype=float),
        np.array(sequence_identities, dtype=float),
    )


def mix_with_white(
    color: str,
    white_fraction: float,
) -> tuple[float, float, float]:
    """
    Create a lighter version of a color by mixing it with white.

    white_fraction:
        0 gives the original color.
        1 gives pure white.
    """
    red, green, blue = to_rgb(color)

    return (
        red + (1.0 - red) * white_fraction,
        green + (1.0 - green) * white_fraction,
        blue + (1.0 - blue) * white_fraction,
    )


def darken_color(
    color: str,
    factor: float = 0.65,
) -> tuple[float, float, float]:
    """
    Create a darker version of a color.
    """
    red, green, blue = to_rgb(color)

    return (
        red * factor,
        green * factor,
        blue * factor,
    )


def create_colormap(
    color: str,
) -> LinearSegmentedColormap:
    """
    Create a visible light-to-dark density colormap.

    The lightest color is deliberately not white. This ensures that
    hexagons containing only one alignment remain visible.
    """
    light_color = mix_with_white(
        color=color,
        white_fraction=0.72,
    )

    dark_color = darken_color(
        color=color,
        factor=0.62,
    )

    return LinearSegmentedColormap.from_list(
        "hexbin_colormap",
        [
            light_color,
            color,
            dark_color,
        ],
    )


def print_summary(
    alignment_lengths: np.ndarray,
    sequence_identities: np.ndarray,
    long_alignment_cutoff: float | None,
) -> None:
    """
    Print summary statistics for the MMseqs alignments.
    """
    print("\nSummary")
    print("-" * 52)

    print(
        f"Number of alignments:      "
        f"{len(alignment_lengths):,}"
    )

    print(
        f"Mean alignment length:     "
        f"{np.mean(alignment_lengths):.1f}"
    )

    print(
        f"Median alignment length:   "
        f"{np.median(alignment_lengths):.1f}"
    )

    print(
        f"Minimum alignment length:  "
        f"{np.min(alignment_lengths):.0f}"
    )

    print(
        f"Maximum alignment length:  "
        f"{np.max(alignment_lengths):.0f}"
    )

    print(
        f"Mean sequence identity:    "
        f"{np.mean(sequence_identities):.3f}"
    )

    print(
        f"Median sequence identity:  "
        f"{np.median(sequence_identities):.3f}"
    )

    print(
        f"Minimum sequence identity: "
        f"{np.min(sequence_identities):.3f}"
    )

    print(
        f"Maximum sequence identity: "
        f"{np.max(sequence_identities):.3f}"
    )

    if long_alignment_cutoff is not None:
        long_count = int(
            np.sum(
                alignment_lengths > long_alignment_cutoff
            )
        )

        print(
            f"Alignments > "
            f"{long_alignment_cutoff:g}:"
            f"{long_count:>16,}"
        )


def make_plot(
    alignment_lengths: np.ndarray,
    sequence_identities: np.ndarray,
    output_prefix: Path,
    color: str,
    title: str,
    x_max: float,
    gridsize: int,
    histogram_bins: int,
    highlight_above: float | None,
) -> None:
    """
    Create a hexbin plot with top and right marginal histograms.

    A faint scatter layer is drawn below the hexbin plot so that every
    observation remains visible, including isolated alignments.
    """
    figure = plt.figure(
        figsize=(5.6, 5.2),
        facecolor="white",
    )

    grid = figure.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1, 4],
        width_ratios=[4, 1],
        hspace=0.04,
        wspace=0.04,
    )

    top_axis = figure.add_subplot(
        grid[0, 0]
    )

    main_axis = figure.add_subplot(
        grid[1, 0],
        sharex=top_axis,
    )

    right_axis = figure.add_subplot(
        grid[1, 1],
        sharey=main_axis,
    )

    colormap = create_colormap(color)

    # Faint scatter layer.
    #
    # This guarantees that isolated observations remain visible even
    # when they fall into hexagons containing only one alignment.
    main_axis.scatter(
        alignment_lengths,
        sequence_identities,
        s=11,
        color=color,
        alpha=0.22,
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )

    # Main density plot.
    main_axis.hexbin(
        alignment_lengths,
        sequence_identities,
        gridsize=gridsize,
        extent=[
            0,
            x_max,
            0,
            1,
        ],
        mincnt=1,
        bins="log",
        cmap=colormap,
        linewidths=0.25,
        edgecolors="white",
        alpha=0.92,
        rasterized=True,
        zorder=2,
    )

    # Optionally outline long alignments.
    if highlight_above is not None:
        long_alignment_mask = (
            alignment_lengths > highlight_above
        )

        if np.any(long_alignment_mask):
            main_axis.scatter(
                alignment_lengths[long_alignment_mask],
                sequence_identities[long_alignment_mask],
                s=32,
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )

    # Top histogram: alignment lengths.
    top_axis.hist(
        alignment_lengths,
        bins=np.linspace(
            0,
            x_max,
            histogram_bins + 1,
        ),
        color=color,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.8,
        rwidth=0.98,
    )

    # Right histogram: sequence identities.
    right_axis.hist(
        sequence_identities,
        bins=np.linspace(
            0,
            1,
            histogram_bins + 1,
        ),
        orientation="horizontal",
        color=color,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.8,
        rwidth=0.98,
    )

    main_axis.set_xlim(
        0,
        x_max,
    )

    main_axis.set_ylim(
        0,
        1.02,
    )

    # Six evenly spaced ticks, including 0 and xmax.
    x_ticks = np.linspace(
        0,
        x_max,
        6,
    )

    main_axis.set_xticks(
        x_ticks
    )

    main_axis.set_xticklabels(
        [
            f"{int(value):d}"
            for value in x_ticks
        ]
    )

    y_ticks = np.arange(
        0,
        1.01,
        0.2,
    )

    main_axis.set_yticks(
        y_ticks
    )

    main_axis.set_yticklabels(
        [
            f"{value:.1f}"
            for value in y_ticks
        ]
    )

    main_axis.set_xlabel(
        "Length of the alignment",
        fontsize=12,
        fontweight="bold",
        labelpad=8,
    )

    main_axis.set_ylabel(
        "Sequence identity",
        fontsize=12,
        fontweight="bold",
        labelpad=8,
    )

    main_axis.tick_params(
        axis="both",
        direction="out",
        length=3.5,
        width=0.9,
        labelsize=10,
    )

    for spine in main_axis.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")

    main_axis.grid(False)

    top_axis.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        pad=7,
    )

    # Remove axes from the top histogram.
    top_axis.tick_params(
        axis="both",
        which="both",
        left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False,
    )

    for spine in top_axis.spines.values():
        spine.set_visible(False)

    # Remove axes from the right histogram.
    right_axis.tick_params(
        axis="both",
        which="both",
        left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False,
    )

    for spine in right_axis.spines.values():
        spine.set_visible(False)

    figure.subplots_adjust(
        left=0.17,
        right=0.94,
        bottom=0.15,
        top=0.88,
    )

    png_file = output_prefix.with_suffix(".png")
    pdf_file = output_prefix.with_suffix(".pdf")
    svg_file = output_prefix.with_suffix(".svg")

    figure.savefig(
        png_file,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        pdf_file,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        svg_file,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)

    print_summary(
        alignment_lengths=alignment_lengths,
        sequence_identities=sequence_identities,
        long_alignment_cutoff=highlight_above,
    )

    print(f"\nSaved plot: {png_file}")
    print(f"Saved plot: {pdf_file}")
    print(f"Saved plot: {svg_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a hexbin plot of MMseqs sequence identity "
            "versus alignment length."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="MMseqs alignment output file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mmseqs_hexbin"),
        help=(
            "Output filename prefix. "
            "Default: mmseqs_hexbin"
        ),
    )

    parser.add_argument(
        "--title",
        type=str,
        default=">10,000 sequences per reaction",
        help="Title shown above the plot.",
    )

    parser.add_argument(
        "--color",
        type=str,
        default="#68AAA3",
        help=(
            "Color used for the hexbin and histograms. "
            "Default: #68AAA3"
        ),
    )

    parser.add_argument(
        "--xmax",
        type=float,
        default=1000,
        help=(
            "Maximum alignment length shown on the x-axis. "
            "Default: 1000"
        ),
    )

    parser.add_argument(
        "--gridsize",
        type=int,
        default=20,
        help=(
            "Approximate number of hexagons along the x-axis. "
            "Default: 20"
        ),
    )

    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=20,
        help=(
            "Number of bins in each marginal histogram. "
            "Default: 20"
        ),
    )

    parser.add_argument(
        "--highlight-above",
        type=float,
        default=None,
        help=(
            "Optionally outline alignments longer than this value. "
            "For example: --highlight-above 800"
        ),
    )

    args = parser.parse_args()

    input_file = args.input.expanduser().resolve()

    if not input_file.is_file():
        raise SystemExit(
            f"Input file does not exist: {input_file}"
        )

    if args.xmax <= 0:
        raise SystemExit(
            "--xmax must be greater than zero"
        )

    if args.gridsize <= 0:
        raise SystemExit(
            "--gridsize must be greater than zero"
        )

    if args.histogram_bins <= 0:
        raise SystemExit(
            "--histogram-bins must be greater than zero"
        )

    output_prefix = args.output.expanduser()

    if not output_prefix.is_absolute():
        output_prefix = Path.cwd() / output_prefix

    output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        alignment_lengths, sequence_identities = (
            read_mmseqs_file(input_file)
        )

    except ValueError as error:
        raise SystemExit(str(error)) from error

    make_plot(
        alignment_lengths=alignment_lengths,
        sequence_identities=sequence_identities,
        output_prefix=output_prefix,
        color=args.color,
        title=args.title,
        x_max=args.xmax,
        gridsize=args.gridsize,
        histogram_bins=args.histogram_bins,
        highlight_above=args.highlight_above,
    )


if __name__ == "__main__":
    main()