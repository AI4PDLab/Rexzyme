"""Plot mean ESMFold pLDDT for sequences generated from each reaction group.

Each group corresponds to a reaction list:

    generation_most     (>10,000 sequences / reaction)
    generation_middle   (100-10,000)
    generation_low      (<100)
    generation_val_com  (0, same EC in training)
    generation_val_dif  (0, same EC not in training)

Expected layout under --root:

    fold/
      generation_most/
        *.pdb
      generation_middle/
        *.pdb
      generation_low/
        *.pdb
      generation_val_com/
        *.pdb
      generation_val_dif/
        *.pdb

For example:

    fold/generation_most/
      top_output_topk100_file_0.fasta_0_ppl_18.82.pdb
      top_output_topk100_file_0.fasta_1_ppl_12.25.pdb
      ...
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GROUPS = [
    {
        "directory": "generation_most",
        "label": ">10,000",
        "display_label": ">10,000",
        "name": "most",
        "color": "#68AAA3",
    },
    {
        "directory": "generation_middle",
        "label": "10,000–100",
        "display_label": "10,000–100",
        "name": "middle",
        "color": "#C9347E",
    },
    {
        "directory": "generation_low",
        "label": "<100",
        "display_label": "<100",
        "name": "low",
        "color": "#F1B548",
    },
    {
        "directory": "generation_val_com",
        "label": "0",
        "display_label": "0\nsame EC in training",
        "name": "val_com",
        "color": "#7396E8",
    },
    {
        "directory": "generation_val_dif",
        "label": "0",
        "display_label": "0\nsame EC not in training",
        "name": "val_dif",
        "color": "#7657D6",
    },
]


def mean_plddt_from_pdb(pdb_path: Path) -> float:
    """
    Calculate the mean residue pLDDT from C-alpha atoms.

    ESMFold stores residue-level pLDDT values in the B-factor column.
    Only C-alpha atoms are used so that each residue contributes once.
    """
    ca_plddt: list[float] = []

    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()

            if atom_name != "CA":
                continue

            try:
                bfactor = float(line[60:66])
            except (ValueError, IndexError):
                continue

            ca_plddt.append(bfactor)

    if not ca_plddt:
        raise ValueError("No valid CA-atom pLDDT values found")

    return float(np.mean(ca_plddt))


def convert_plddt_to_percent(mean_plddt: float) -> float:
    """
    Convert pLDDT to a 0-100 scale.

    Some ESMFold PDB files store pLDDT between 0 and 1, while others
    store it directly between 0 and 100.
    """
    if 0 <= mean_plddt <= 1.5:
        return mean_plddt * 100

    return mean_plddt


def find_pdb_files(group_directory: Path) -> list[Path]:
    """
    Find PDB files directly inside one fold/generation_* directory.

    Expected structure:

        fold/
          generation_most/
            *.pdb
          generation_middle/
            *.pdb
          generation_low/
            *.pdb
          generation_val_com/
            *.pdb
          generation_val_dif/
            *.pdb
    """
    return sorted(group_directory.glob("*.pdb"))


def collect_plddt(fold_directory: Path) -> list[dict[str, object]]:
    """
    Extract the mean pLDDT of every PDB file in each experimental group.

    The PDB directories are expected under:

        <fold_directory>/generation_*
    """
    records: list[dict[str, object]] = []

    if not fold_directory.is_dir():
        raise SystemExit(
            f"Fold directory does not exist: {fold_directory}"
        )

    for group in GROUPS:
        group_directory = fold_directory / str(group["directory"])

        if not group_directory.is_dir():
            print(f"Warning: directory not found: {group_directory}")
            continue

        pdb_files = find_pdb_files(group_directory)

        print(
            f"{group['name']}: found {len(pdb_files):,} PDB files "
            f"in {group_directory}"
        )

        for pdb_path in pdb_files:
            try:
                raw_mean_plddt = mean_plddt_from_pdb(pdb_path)
                mean_plddt_percent = convert_plddt_to_percent(
                    raw_mean_plddt
                )

            except (OSError, ValueError) as error:
                print(f"Warning: skipped {pdb_path}: {error}")
                continue

            records.append(
                {
                    "group": group["name"],
                    "label": group["label"],
                    "directory": group["directory"],
                    "pdb": str(pdb_path.relative_to(root)),
                    "mean_plddt": mean_plddt_percent,
                }
            )

    return records


def write_tsv(
    records: list[dict[str, object]],
    output_path: Path,
) -> None:
    """
    Save all extracted pLDDT values to a TSV file.
    """
    fieldnames = [
        "group",
        "label",
        "directory",
        "pdb",
        "mean_plddt",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
        )

        writer.writeheader()
        writer.writerows(records)


def print_summary(
    records: list[dict[str, object]],
    confidence_threshold: float,
    expected_total: int = 500,
) -> None:
    """
    Print descriptive statistics for each group.

    Percent:
        Percentage of available structures with pLDDT >= threshold.

    N/500 (%):
        Percentage of the expected 500 structures that were found.
    """
    print("\nSummary")
    print("-" * 90)

    print(
        f"{'Group':<15}"
        f"{'N':>8}"
        f"{'N/500 (%)':>12}"
        f"{'Mean':>10}"
        f"{'Median':>10}"
        f"{f'>={confidence_threshold:g}':>12}"
        f"{'Percent':>12}"
    )

    for group in GROUPS:
        values = np.array(
            [
                float(record["mean_plddt"])
                for record in records
                if record["group"] == group["name"]
            ],
            dtype=float,
        )

        if len(values) == 0:
            continue

        number_structures = len(values)

        percentage_of_expected = (
            100 * number_structures / expected_total
        )

        high_count = int(
            np.sum(values >= confidence_threshold)
        )

        high_percent = (
            100 * high_count / number_structures
        )

        print(
            f"{str(group['name']):<15}"
            f"{number_structures:>8,}"
            f"{percentage_of_expected:>11.1f}%"
            f"{np.mean(values):>10.2f}"
            f"{np.median(values):>10.2f}"
            f"{high_count:>12,}"
            f"{high_percent:>11.1f}%"
        )


def make_plot(
    records: list[dict[str, object]],
    output_prefix: Path,
    confidence_threshold: float = 70,
    seed: int = 42,
) -> None:
    """
    Create a jittered dot plot of mean ESMFold pLDDT values.
    """
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(7.0, 5.6))

    group_counts: list[int] = []

    for x_position, group in enumerate(GROUPS):
        values = np.array(
            [
                float(record["mean_plddt"])
                for record in records
                if record["group"] == group["name"]
            ],
            dtype=float,
        )

        group_counts.append(len(values))

        if len(values) == 0:
            continue

        jitter = rng.uniform(-0.24, 0.24, size=len(values))

        x_values = x_position + jitter

        low_confidence = values < confidence_threshold
        high_confidence = values >= confidence_threshold

        # Low-confidence structures.
        ax.scatter(
            x_values[low_confidence],
            values[low_confidence],
            s=18,
            c="#7A7A7A",
            alpha=0.82,
            edgecolors="#666666",
            linewidths=0.25,
            rasterized=True,
            zorder=2,
        )

        # High-confidence structures.
        ax.scatter(
            x_values[high_confidence],
            values[high_confidence],
            s=20,
            c=str(group["color"]),
            alpha=0.92,
            edgecolors=str(group["color"]),
            linewidths=0.45,
            rasterized=True,
            zorder=4,
        )

    # Dashed line at the high-confidence threshold.
    ax.axhline(
        y=confidence_threshold,
        color="#B5B5B5",
        linestyle=(0, (4, 4)),
        linewidth=0.9,
        alpha=0.8,
        zorder=1,
    )

    ax.set_xlim(-0.5, len(GROUPS) - 0.5)
    ax.set_ylim(15, 100)

    ax.set_yticks(np.arange(0, 101, 20))

    tick_labels = [
        f"{group['display_label']}\n$n$ = {count:,}"
        for group, count in zip(GROUPS, group_counts)
    ]

    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels(
        tick_labels,
        fontsize=8.5,
    )

    ax.set_ylabel(
        "Mean ESMFold pLDDT (%)",
        fontsize=12,
        fontweight="semibold",
        labelpad=8,
    )

    ax.set_xlabel(
        "Number of sequences per reaction in the training data",
        fontsize=11,
        fontweight="semibold",
        labelpad=10,
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=9,
        direction="out",
        length=3.5,
        width=0.9,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.set_axisbelow(True)

    ax.yaxis.grid(
        True,
        linewidth=0.5,
        alpha=0.16,
    )

    ax.xaxis.grid(False)

    ax.text(
        -0.105,
        1.025,
        "A",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    fig.tight_layout(pad=1.0)

    fig.subplots_adjust(bottom=0.25)

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    svg_path = output_prefix.with_suffix(".svg")

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"\nSaved plot: {png_path}")
    print(f"Saved plot: {pdf_path}")
    print(f"Saved plot: {svg_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean ESMFold pLDDT for generation groups."
        )
    )

    parser.add_argument(
        "--fold-dir",
        "--input",
        dest="fold_dir",
        type=Path,
        default=None,
        help=(
            "Directory containing generation_most, generation_middle, "
            "generation_low, generation_val_com, and generation_val_dif "
            "PDB folders. Overrides --root/fold."
        ),
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help=(
            "Repository root containing the fold/ directory when "
            "--fold-dir is not set. Default: current directory."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=70,
        help=(
            "pLDDT threshold above which points are colored. "
            "Default: 70."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plddt_by_training_group"),
        help=(
            "Output filename prefix. "
            "Default: plddt_by_training_group."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed used for horizontal point jitter. "
            "Default: 42."
        ),
    )

    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    fold_directory = (
        args.fold_dir.expanduser().resolve()
        if args.fold_dir is not None
        else root / "fold"
    )

    if args.fold_dir is None and not root.is_dir():
        raise SystemExit(
            f"Root directory does not exist: {root}"
        )

    records = collect_plddt(fold_directory)

    if not records:
        raise SystemExit(
            "No pLDDT values were extracted. Expected PDB files under "
            "<fold-dir>/generation_*/*.pdb "
            "(or <root>/fold/generation_*/*.pdb)."
        )

    output_prefix = args.output.expanduser()

    output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tsv_path = output_prefix.with_suffix(".tsv")

    write_tsv(
        records,
        tsv_path,
    )

    print(f"\nSaved extracted values: {tsv_path}")

    print_summary(
        records,
        confidence_threshold=args.threshold,
    )

    make_plot(
        records,
        output_prefix=output_prefix,
        confidence_threshold=args.threshold,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
