"""Shared helpers for held-out TM-score scatter plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from scipy.stats import pearsonr, spearmanr

PAPER_COLORS = {
    "most": "#68AAA3",
    "middle": "#C9347E",
    "low": "#F1B548",
    "val_com": "#7396E8",
    "val_dif": "#7657D6",
}

DENSITY_CMAPS = {
    "most": LinearSegmentedColormap.from_list(
        "most_density",
        ["#EFF7F6", "#CFE7E4", "#9ACBC6", "#68AAA3", "#397D77", "#184F4B"],
    ),
    "middle": LinearSegmentedColormap.from_list(
        "middle_density",
        ["#FBEFF5", "#F2CADD", "#E49ABD", "#C9347E", "#8E2057", "#591036"],
    ),
    "low": LinearSegmentedColormap.from_list(
        "low_density",
        ["#FFF8EA", "#FBE8C2", "#F6D384", "#F1B548", "#C38319", "#79500B"],
    ),
    "val_com": LinearSegmentedColormap.from_list(
        "val_com_density",
        ["#F0F4FD", "#D5DFF9", "#A9BDF2", "#7396E8", "#4267B9", "#243D7D"],
    ),
    "val_dif": LinearSegmentedColormap.from_list(
        "val_dif_density",
        ["#F3F0FC", "#DBD4F7", "#AFA0EA", "#7657D6", "#5039A7", "#30226D"],
    ),
}

VAL_DIF_COLOR = PAPER_COLORS["val_dif"]
VAL_DIF_DARK = "#3F2A88"
LINEAR_FIT_COLOR = "black"

METRIC_LABELS = {
    "ttmscore": "target-normalized TM-score",
    "qtmscore": "query-normalized TM-score",
    "alntmscore": "alignment TM-score",
}

TM_METRIC_CHOICES = list(METRIC_LABELS)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def reaction_number(reaction: str) -> int | None:
    try:
        return int(str(reaction).replace("output_file_", "").replace("output_file-", ""))
    except ValueError:
        return None


def design_stem(design: str) -> str:
    name = Path(str(design)).name
    if name.endswith(".pdb") or name.endswith(".cif"):
        name = name.rsplit(".", 1)[0]
    return name


def required_columns(header: list[str], names: set[str], path) -> dict[str, int]:
    columns = {name: index for index, name in enumerate(header)}
    missing = names.difference(columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(sorted(missing))}")
    return columns


def read_best_tmscores(master_path: Path, tm_metric: str) -> dict[tuple[str, str], float]:
    best: dict[tuple[str, str], float] = {}
    with master_path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        columns = required_columns(header, {"reaction", "design", tm_metric}, master_path)
        max_index = max(columns["reaction"], columns["design"], columns[tm_metric])
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= max_index:
                continue
            try:
                tm_score = float(fields[columns[tm_metric]])
            except ValueError:
                continue
            if not np.isfinite(tm_score):
                continue
            key = (fields[columns["reaction"]], design_stem(fields[columns["design"]]))
            if key not in best or tm_score > best[key]:
                best[key] = tm_score
    return best


def retain_best_design_per_reaction(
    best: dict[tuple[str, str], float],
) -> dict[tuple[str, str], float]:
    per_reaction: dict[str, tuple[str, float]] = {}
    for (reaction, design), tm_score in best.items():
        if reaction not in per_reaction or tm_score > per_reaction[reaction][1]:
            per_reaction[reaction] = (design, tm_score)
    return {
        (reaction, design): tm_score
        for reaction, (design, tm_score) in per_reaction.items()
    }


def read_best_designs_after_plddt_filter(
    plddt_tsv: Path,
    tm_metric: str,
    minimum_plddt: float,
) -> dict[str, dict[str, object]]:
    """Keep designs with pLDDT >= threshold, then the best TM-score per reaction."""
    tm_column = f"best_{tm_metric}"
    selected: dict[str, dict[str, object]] = {}
    total = passing = 0
    with plddt_tsv.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        columns = required_columns(
            header, {"reaction", "design", "mean_plddt", tm_column}, plddt_tsv
        )
        max_index = max(
            columns["reaction"], columns["design"], columns["mean_plddt"], columns[tm_column]
        )
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= max_index:
                continue
            total += 1
            try:
                plddt = float(fields[columns["mean_plddt"]])
                tm_score = float(fields[columns[tm_column]])
            except ValueError:
                continue
            if not np.isfinite(plddt) or not np.isfinite(tm_score):
                continue
            if plddt < minimum_plddt:
                continue
            passing += 1
            reaction = fields[columns["reaction"]]
            if reaction not in selected or tm_score > selected[reaction]["tm_score"]:
                selected[reaction] = {
                    "design": design_stem(fields[columns["design"]]),
                    "plddt": plddt,
                    "tm_score": tm_score,
                }
    print(f"Designs read: {total:,}")
    print(f"Designs with pLDDT >= {minimum_plddt:g}: {passing:,}")
    print(f"Reactions with at least one design passing the filter: {len(selected):,}")
    return selected


def mean_plddt(pdb_path: Path) -> float | None:
    values = []
    try:
        with pdb_path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith("ATOM"):
                    continue
                if line[12:16].strip() != "CA":
                    continue
                try:
                    values.append(float(line[60:66]))
                except ValueError:
                    continue
    except FileNotFoundError:
        return None
    if not values:
        return None
    return float(np.mean(values))


def resolve_design_pdb(designs_root: Path, reaction: str, design: str) -> Path | None:
    stem = design_stem(design)
    number = reaction_number(reaction)
    candidates = []
    as_path = Path(design)
    if as_path.is_file():
        candidates.append(as_path)
    if number is not None:
        candidates.extend(
            [
                designs_root / f"output_file_{number}" / "pdbs" / f"{stem}.pdb",
                designs_root / f"output_file_{number}" / f"{stem}.pdb",
                designs_root / f"output_file-{number}" / "pdbs" / f"{stem}.pdb",
                designs_root / f"output_file-{number}" / f"{stem}.pdb",
            ]
        )
    candidates.append(designs_root / f"{stem}.pdb")
    for path in candidates:
        if path.is_file():
            return path
    return None


def correlations_and_fit(xs: np.ndarray, ys: np.ndarray) -> dict[str, float]:
    pearson_r, pearson_p = pearsonr(xs, ys)
    spearman_rho, spearman_p = spearmanr(xs, ys)
    slope, intercept = np.polyfit(xs, ys, deg=1)
    r_squared = float(pearson_r) ** 2
    print(f"Pearson r = {pearson_r:.3f}, p = {pearson_p:.3e}")
    print(f"Spearman rho = {spearman_rho:.3f}, p = {spearman_p:.3e}")
    print(f"Linear fit: y = {slope:.5f}x {intercept:+.5f}; R^2 = {r_squared:.3f}")
    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
    }


def plot_linear_fit(ax, xs: np.ndarray, fit: dict[str, float]) -> None:
    fit_x = np.linspace(xs.min(), xs.max(), 300)
    ax.plot(
        fit_x,
        fit["slope"] * fit_x + fit["intercept"],
        color=LINEAR_FIT_COLOR,
        linestyle="--",
        linewidth=2.2,
        alpha=0.90,
        zorder=10,
        label="Linear fit",
    )


def configure_tm_axes(ax, x_max: float = 100, x_min: float = 0) -> None:
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(axis="both", which="major", length=5, width=0.8)
    ax.tick_params(axis="both", which="minor", length=3, width=0.6)
    ax.set_axisbelow(True)
    ax.grid(which="major", linewidth=0.7, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, out_prefix: Path, dpi: int) -> None:
    png_path = Path(f"{out_prefix}.png")
    pdf_path = Path(f"{out_prefix}.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved PNG:  {png_path}")
    print(f"Saved PDF:  {pdf_path}")


def scatter_heldout(ax, xs, ys, point_size=85, point_alpha=0.62, edge_width=0.7) -> None:
    ax.scatter(
        xs,
        ys,
        s=point_size,
        facecolors=VAL_DIF_COLOR,
        edgecolors=VAL_DIF_DARK,
        linewidths=edge_width,
        alpha=point_alpha,
        rasterized=True,
        zorder=2,
    )
