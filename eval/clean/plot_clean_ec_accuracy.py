"""Plot CLEAN EC-label accuracy at hierarchy levels 1–4.

Prediction and reference EC files can be supplied explicitly for any
subset of the five reaction groups:

    most
    middle
    low
    val_com
    val_dif

For each group that should be evaluated, provide BOTH:

    1. CLEAN prediction file
    2. Ground-truth/reference EC file

Example:

    python clean_ec_accuracy.py \
        --most-csv path/to/most_merged_output.csv \
        --most-ec path/to/most_smiles_with_ec.tsv

If only one group is provided, only that group is plotted.

If all five groups are provided, all five are plotted.

CLEAN itself is an external dependency and is not run by this script.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# GROUP DEFINITIONS
# ---------------------------------------------------------------------

GROUPS = [
    {
        "key": "most",
        "name": "most",
        "label": ">10,000",
        "csv_argument": "most_csv",
        "ec_argument": "most_ec",
    },
    {
        "key": "middle",
        "name": "middle",
        "label": "10,000–100",
        "csv_argument": "middle_csv",
        "ec_argument": "middle_ec",
    },
    {
        "key": "low",
        "name": "low",
        "label": "<100",
        "csv_argument": "low_csv",
        "ec_argument": "low_ec",
    },
    {
        "key": "val_com",
        "name": "val_com",
        "label": "0\nsame EC in training",
        "csv_argument": "val_com_csv",
        "ec_argument": "val_com_ec",
    },
    {
        "key": "val_dif",
        "name": "val_dif",
        "label": "0\nsame EC not in training",
        "csv_argument": "val_dif_csv",
        "ec_argument": "val_dif_ec",
    },
]


# ---------------------------------------------------------------------
# EC LEVEL DEFINITIONS
# ---------------------------------------------------------------------

EC_LEVELS = [
    {
        "level": 1,
        "label": "EC X.-.-.-",
        "color": "#6489F5",
    },
    {
        "level": 2,
        "label": "EC X.X.-.-",
        "color": "#C93780",
    },
    {
        "level": 3,
        "label": "EC X.X.X.-",
        "color": "#EB6428",
    },
    {
        "level": 4,
        "label": "EC X.X.X.X",
        "color": "#F5AE32",
    },
]


# ---------------------------------------------------------------------
# CLEAN FILE FORMAT
# ---------------------------------------------------------------------

SAMPLE_ID_PATTERN = re.compile(
    r"^(\d+)_(\d+)$"
)

PREDICTION_PATTERN = re.compile(
    r"EC:([^/,\s]+)(?:/([^,\s]+))?"
)


# ---------------------------------------------------------------------
# GROUND TRUTH
# ---------------------------------------------------------------------

def split_ec_labels(
    ec_field: str,
) -> list[str]:
    """
    Split one or more ground-truth EC labels.

    Multiple labels may be separated using semicolons.

    Examples:

        1.1.1.1

        EC:1.1.1.1

        1.1.1.1;1.1.1.2
    """

    ec_field = ec_field.strip()

    if not ec_field:
        return []

    if ec_field.lower() == "not found":
        return []

    ec_labels: list[str] = []

    for value in ec_field.split(";"):

        value = value.strip()

        if value.startswith("EC:"):
            value = value[3:]

        if (
            value
            and value.lower() != "not found"
        ):
            ec_labels.append(value)

    return ec_labels


def read_ground_truth(
    ground_truth_file: Path,
) -> list[dict[str, object]]:
    """
    Read the ground-truth EC file.

    The row position defines the reaction index:

        first data row  -> reaction 0
        second data row -> reaction 1
        third data row  -> reaction 2

    The first line is assumed to be a header.

    For tab-separated files:

        SMILES    EC

    the first column is taken as the reaction SMILES and the final
    column is taken as the EC label.
    """

    records: list[dict[str, object]] = []

    with ground_truth_file.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        header = handle.readline()

        if not header:
            raise ValueError(
                f"Ground-truth file is empty: "
                f"{ground_truth_file}"
            )

        for reaction_index, line in enumerate(handle):

            line = line.rstrip("\n\r")

            if not line.strip():
                continue

            if "\t" in line:

                fields = line.split("\t")

                smiles = fields[0].strip()
                ec_field = fields[-1].strip()

            else:

                fields = line.split(maxsplit=1)

                if len(fields) < 2:
                    raise ValueError(
                        f"Malformed ground-truth line for "
                        f"reaction {reaction_index}: {line}"
                    )

                smiles = fields[0].strip()
                ec_field = fields[1].strip()

            records.append(
                {
                    "reaction": reaction_index,
                    "smiles": smiles,
                    "ground_truth_ec": split_ec_labels(
                        ec_field
                    ),
                    "ground_truth_ec_text": ec_field,
                }
            )

    return records


# ---------------------------------------------------------------------
# CLEAN PREDICTIONS
# ---------------------------------------------------------------------

def parse_prediction_entries(
    prediction_text: str,
) -> list[tuple[str, float | None]]:
    """
    Parse CLEAN predictions.

    Example:

        EC:3.2.2.4/0.0007,EC:3.2.2.n1/0.0006

    Returns:

        [
            ("3.2.2.4", 0.0007),
            ("3.2.2.n1", 0.0006),
        ]
    """

    predictions: list[
        tuple[str, float | None]
    ] = []

    for match in PREDICTION_PATTERN.finditer(
        prediction_text
    ):

        ec_label = match.group(1).strip()

        score_text = match.group(2)

        score: float | None = None

        if score_text is not None:

            try:
                score = float(score_text)

            except ValueError:
                score = None

        predictions.append(
            (
                ec_label,
                score,
            )
        )

    return predictions


def select_predictions(
    predictions: list[
        tuple[str, float | None]
    ],
    prediction_mode: str,
) -> list[str]:
    """
    Select which CLEAN predictions to evaluate.

    any
        A sequence is considered correct if ANY CLEAN prediction
        matches the reference EC.

    first
        Only the first CLEAN prediction is evaluated.

    lowest-score
        Only the CLEAN prediction with the lowest numerical score
        is evaluated.
    """

    if not predictions:
        return []

    if prediction_mode == "any":

        return [
            ec_label
            for ec_label, _ in predictions
        ]

    if prediction_mode == "first":

        return [
            predictions[0][0]
        ]

    if prediction_mode == "lowest-score":

        predictions_with_scores = [
            prediction
            for prediction in predictions
            if prediction[1] is not None
        ]

        if not predictions_with_scores:

            return [
                predictions[0][0]
            ]

        best_prediction = min(
            predictions_with_scores,
            key=lambda item: float(item[1]),
        )

        return [
            best_prediction[0]
        ]

    raise ValueError(
        f"Unsupported prediction mode: "
        f"{prediction_mode}"
    )


def read_predictions(
    prediction_file: Path,
    prediction_mode: str,
) -> dict[
    int,
    list[dict[str, object]],
]:
    """
    Read CLEAN predictions and group sequences by reaction index.

    Expected identifiers:

        0_0
        0_1
        0_2
        1_0
        1_1

    where:

        first number  = reaction index
        second number = sequence index
    """

    predictions_by_reaction: dict[
        int,
        list[dict[str, object]],
    ] = {}

    seen_sample_ids: set[str] = set()

    with prediction_file.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        for line_number, line in enumerate(
            handle,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            fields = line.split(
                maxsplit=1
            )

            sample_id = fields[0]

            match = SAMPLE_ID_PATTERN.fullmatch(
                sample_id
            )

            if match is None:

                raise ValueError(
                    f"Unexpected sample identifier at "
                    f"{prediction_file}:{line_number}: "
                    f"{sample_id}"
                )

            if sample_id in seen_sample_ids:

                raise ValueError(
                    f"Duplicate sample identifier in "
                    f"{prediction_file}: {sample_id}"
                )

            seen_sample_ids.add(
                sample_id
            )

            reaction_index = int(
                match.group(1)
            )

            sequence_index = int(
                match.group(2)
            )

            prediction_text = (
                fields[1]
                if len(fields) > 1
                else ""
            )

            parsed_predictions = (
                parse_prediction_entries(
                    prediction_text
                )
            )

            selected_ec_labels = (
                select_predictions(
                    parsed_predictions,
                    prediction_mode=prediction_mode,
                )
            )

            predictions_by_reaction.setdefault(
                reaction_index,
                [],
            ).append(
                {
                    "sample_id": sample_id,
                    "reaction": reaction_index,
                    "sequence": sequence_index,
                    "predicted_ec": selected_ec_labels,
                    "raw_prediction": prediction_text,
                }
            )

    for reaction_predictions in (
        predictions_by_reaction.values()
    ):

        reaction_predictions.sort(
            key=lambda record: int(
                record["sequence"]
            )
        )

    return predictions_by_reaction


# ---------------------------------------------------------------------
# EC COMPARISON
# ---------------------------------------------------------------------

def ec_parts(
    ec_label: str,
) -> tuple[str, ...]:
    """
    Convert an EC label into hierarchical components.

    Example:

        1.1.1.1

    becomes:

        ("1", "1", "1", "1")
    """

    ec_label = ec_label.strip()

    if ec_label.startswith("EC:"):
        ec_label = ec_label[3:]

    return tuple(
        part.strip()
        for part in ec_label.split(".")
    )


def ground_truth_valid_at_level(
    ground_truth_ec: str,
    level: int,
) -> bool:
    """
    Return whether a ground-truth EC can be evaluated
    at the requested hierarchy level.
    """

    parts = ec_parts(
        ground_truth_ec
    )

    if len(parts) < level:
        return False

    invalid_parts = {
        "",
        "-",
        "?",
        "NA",
        "N/A",
    }

    return all(
        part.upper() not in invalid_parts
        for part in parts[:level]
    )


def ec_matches_at_level(
    predicted_ec: str,
    ground_truth_ec: str,
    level: int,
) -> bool:
    """
    Compare predicted and reference EC labels at one level.

    Example:

        prediction:   3.2.2.n1
        ground truth: 3.2.2.4

        level 1 -> correct
        level 2 -> correct
        level 3 -> correct
        level 4 -> incorrect
    """

    predicted_parts = ec_parts(
        predicted_ec
    )

    ground_truth_parts = ec_parts(
        ground_truth_ec
    )

    if (
        len(predicted_parts) < level
        or len(ground_truth_parts) < level
    ):
        return False

    return (
        predicted_parts[:level]
        == ground_truth_parts[:level]
    )


def sequence_is_correct(
    predicted_ec_labels: list[str],
    ground_truth_ec_labels: list[str],
    level: int,
) -> bool:
    """
    Return True if any predicted EC matches any reference EC.
    """

    valid_ground_truth_labels = [
        ec_label
        for ec_label in ground_truth_ec_labels
        if ground_truth_valid_at_level(
            ec_label,
            level,
        )
    ]

    return any(
        ec_matches_at_level(
            predicted_ec,
            ground_truth_ec,
            level,
        )
        for predicted_ec in predicted_ec_labels
        for ground_truth_ec in valid_ground_truth_labels
    )


# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------

def evaluate_group(
    group: dict[str, str],
    prediction_file: Path,
    ground_truth_file: Path,
    prediction_mode: str,
) -> list[dict[str, object]]:
    """
    Count correct sequence predictions for every reaction.
    """

    ground_truth_records = (
        read_ground_truth(
            ground_truth_file
        )
    )

    predictions_by_reaction = (
        read_predictions(
            prediction_file,
            prediction_mode=prediction_mode,
        )
    )

    number_ground_truth_reactions = len(
        ground_truth_records
    )

    unexpected_reactions = sorted(
        reaction
        for reaction in predictions_by_reaction
        if reaction >= number_ground_truth_reactions
    )

    if unexpected_reactions:

        raise ValueError(
            f"{prediction_file} contains reaction "
            f"indices not present in "
            f"{ground_truth_file}: "
            f"{unexpected_reactions}"
        )

    results: list[
        dict[str, object]
    ] = []

    for ground_truth_record in (
        ground_truth_records
    ):

        reaction_index = int(
            ground_truth_record["reaction"]
        )

        ground_truth_ec_labels = list(
            ground_truth_record[
                "ground_truth_ec"
            ]
        )

        reaction_predictions = (
            predictions_by_reaction.get(
                reaction_index,
                [],
            )
        )

        result: dict[
            str,
            object,
        ] = {
            "group": group["name"],
            "group_label": (
                group["label"].replace(
                    "\n",
                    " ",
                )
            ),
            "reaction": reaction_index,
            "ground_truth_ec": (
                ";".join(
                    ground_truth_ec_labels
                )
                if ground_truth_ec_labels
                else "Not Found"
            ),
            "number_sequences": len(
                reaction_predictions
            ),
        }

        for level_definition in (
            EC_LEVELS
        ):

            level = int(
                level_definition["level"]
            )

            evaluable_ground_truth = [
                ec_label
                for ec_label in ground_truth_ec_labels
                if ground_truth_valid_at_level(
                    ec_label,
                    level,
                )
            ]

            if not evaluable_ground_truth:

                result[
                    f"evaluable_level_{level}"
                ] = False

                result[
                    f"correct_level_{level}"
                ] = ""

                result[
                    f"percent_level_{level}"
                ] = ""

                continue

            correct_count = sum(
                sequence_is_correct(
                    list(
                        prediction[
                            "predicted_ec"
                        ]
                    ),
                    ground_truth_ec_labels,
                    level=level,
                )
                for prediction in (
                    reaction_predictions
                )
            )

            number_sequences = len(
                reaction_predictions
            )

            correct_percent = (
                100
                * correct_count
                / number_sequences
                if number_sequences > 0
                else 0.0
            )

            result[
                f"evaluable_level_{level}"
            ] = True

            result[
                f"correct_level_{level}"
            ] = correct_count

            result[
                f"percent_level_{level}"
            ] = correct_percent

        results.append(
            result
        )

    return results


# ---------------------------------------------------------------------
# GROUP SUMMARY
# ---------------------------------------------------------------------

def summarize_group(
    group: dict[str, str],
    reaction_results: list[
        dict[str, object]
    ],
    aggregation: str,
) -> dict[str, object]:
    """
    Summarize per-reaction results.

    micro
        Total correct sequences /
        total evaluated sequences.

    macro
        Mean of the per-reaction percentages.
    """

    summary: dict[
        str,
        object,
    ] = {

        "group": group["name"],

        "group_label": (
            group["label"].replace(
                "\n",
                " ",
            )
        ),

        "number_reactions": len(
            reaction_results
        ),

        "number_sequences": sum(
            int(
                result["number_sequences"]
            )
            for result in reaction_results
        ),
    }

    for level_definition in EC_LEVELS:

        level = int(
            level_definition["level"]
        )

        evaluable_results = [
            result
            for result in reaction_results
            if result[
                f"evaluable_level_{level}"
            ]
        ]

        correct_count = sum(
            int(
                result[
                    f"correct_level_{level}"
                ]
            )
            for result in evaluable_results
        )

        evaluated_sequences = sum(
            int(
                result["number_sequences"]
            )
            for result in evaluable_results
        )

        micro_percent = (
            100
            * correct_count
            / evaluated_sequences
            if evaluated_sequences > 0
            else 0.0
        )

        per_reaction_percentages = [
            float(
                result[
                    f"percent_level_{level}"
                ]
            )
            for result in evaluable_results
            if int(
                result["number_sequences"]
            ) > 0
        ]

        macro_percent = (
            float(
                np.mean(
                    per_reaction_percentages
                )
            )
            if per_reaction_percentages
            else 0.0
        )

        selected_percent = (
            micro_percent
            if aggregation == "micro"
            else macro_percent
        )

        summary[
            f"correct_level_{level}"
        ] = correct_count

        summary[
            f"evaluated_sequences_level_{level}"
        ] = evaluated_sequences

        summary[
            f"micro_percent_level_{level}"
        ] = micro_percent

        summary[
            f"macro_percent_level_{level}"
        ] = macro_percent

        summary[
            f"plot_percent_level_{level}"
        ] = selected_percent

    return summary


# ---------------------------------------------------------------------
# OUTPUT TABLES
# ---------------------------------------------------------------------

def write_per_reaction_results(
    results: list[
        dict[str, object]
    ],
    output_file: Path,
) -> None:
    """
    Write counts and percentages for individual reactions.
    """

    fieldnames = [
        "group",
        "group_label",
        "reaction",
        "ground_truth_ec",
        "number_sequences",
    ]

    for level_definition in EC_LEVELS:

        level = int(
            level_definition["level"]
        )

        fieldnames.extend(
            [
                f"evaluable_level_{level}",
                f"correct_level_{level}",
                f"percent_level_{level}",
            ]
        )

    with output_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()

        writer.writerows(
            results
        )


def write_group_summary(
    summaries: list[
        dict[str, object]
    ],
    output_file: Path,
) -> None:
    """
    Write one summary row for each supplied group.
    """

    fieldnames = [
        "group",
        "group_label",
        "number_reactions",
        "number_sequences",
    ]

    for level_definition in EC_LEVELS:

        level = int(
            level_definition["level"]
        )

        fieldnames.extend(
            [
                f"correct_level_{level}",
                f"evaluated_sequences_level_{level}",
                f"micro_percent_level_{level}",
                f"macro_percent_level_{level}",
                f"plot_percent_level_{level}",
            ]
        )

    with output_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()

        writer.writerows(
            summaries
        )


# ---------------------------------------------------------------------
# PRINT SUMMARY
# ---------------------------------------------------------------------

def print_summary(
    summaries: list[
        dict[str, object]
    ],
    aggregation: str,
) -> None:
    """
    Print EC-level accuracy.
    """

    print(
        f"\nGroup summary "
        f"({aggregation} aggregation)"
    )

    print("-" * 94)

    print(
        f"{'Group':<12}"
        f"{'Reactions':>11}"
        f"{'Sequences':>11}"
        f"{'EC level 1':>15}"
        f"{'EC level 2':>15}"
        f"{'EC level 3':>15}"
        f"{'EC level 4':>15}"
    )

    for summary in summaries:

        print(
            f"{str(summary['group']):<12}"
            f"{int(summary['number_reactions']):>11,}"
            f"{int(summary['number_sequences']):>11,}"
            f"{float(summary['plot_percent_level_1']):>14.1f}%"
            f"{float(summary['plot_percent_level_2']):>14.1f}%"
            f"{float(summary['plot_percent_level_3']):>14.1f}%"
            f"{float(summary['plot_percent_level_4']):>14.1f}%"
        )


# ---------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------

def make_plot(
    summaries: list[
        dict[str, object]
    ],
    active_groups: list[
        dict[str, str]
    ],
    output_prefix: Path,
    aggregation: str,
) -> None:
    """
    Create the EC-level accuracy bar plot.

    Only groups supplied on the command line are plotted.
    """

    number_groups = len(
        active_groups
    )

    # Automatically adapt figure width depending on
    # how many groups are being plotted.
    if number_groups == 1:

        figure_width = 5.0

    elif number_groups == 2:

        figure_width = 6.0

    elif number_groups == 3:

        figure_width = 7.0

    elif number_groups == 4:

        figure_width = 8.0

    else:

        figure_width = 9.2

    figure, axis = plt.subplots(
        figsize=(
            figure_width,
            5.4,
        )
    )

    x_positions = np.arange(
        number_groups
    )

    number_levels = len(
        EC_LEVELS
    )

    total_group_width = 0.72

    bar_width = (
        total_group_width
        / number_levels
    )

    offsets = (
        np.arange(number_levels)
        - (number_levels - 1) / 2
    ) * bar_width

    for (
        offset,
        level_definition,
    ) in zip(
        offsets,
        EC_LEVELS,
    ):

        level = int(
            level_definition["level"]
        )

        values = [
            float(
                summary[
                    f"plot_percent_level_{level}"
                ]
            )
            for summary in summaries
        ]

        axis.bar(
            x_positions + offset,
            values,
            width=bar_width,
            label=str(
                level_definition["label"]
            ),
            color=str(
                level_definition["color"]
            ),
            edgecolor="none",
            zorder=3,
        )

    # Dynamic X-axis limits.
    if number_groups == 1:

        axis.set_xlim(
            -0.6,
            0.6,
        )

    else:

        axis.set_xlim(
            -0.55,
            number_groups - 0.45,
        )

    axis.set_ylim(
        0,
        105,
    )

    axis.set_yticks(
        np.arange(
            0,
            101,
            20,
        )
    )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        [
            group["label"]
            for group in active_groups
        ],
        fontsize=9,
    )

    axis.set_ylabel(
        "Correct EC prediction (%)",
        fontsize=12,
        fontweight="semibold",
        labelpad=8,
    )

    # If only one group is supplied, this long xlabel is
    # not particularly useful, so use a simpler description.
    if number_groups == 1:

        axis.set_xlabel(
            "Reaction group",
            fontsize=11,
            fontweight="semibold",
            labelpad=10,
        )

    else:

        axis.set_xlabel(
            "Number of sequences per reaction in the training data",
            fontsize=11,
            fontweight="semibold",
            labelpad=10,
        )

    axis.tick_params(
        axis="both",
        direction="out",
        length=3.5,
        width=0.9,
        labelsize=9,
    )

    axis.spines[
        "top"
    ].set_visible(
        False
    )

    axis.spines[
        "right"
    ].set_visible(
        False
    )

    axis.spines[
        "left"
    ].set_linewidth(
        1.0
    )

    axis.spines[
        "bottom"
    ].set_linewidth(
        1.0
    )

    axis.set_axisbelow(
        True
    )

    axis.yaxis.grid(
        True,
        linewidth=0.6,
        alpha=0.18,
    )

    axis.xaxis.grid(
        False
    )

    axis.legend(
        title="EC level",
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="#777777",
        fontsize=9,
        title_fontsize=10,
    )

    figure.tight_layout()

    figure.subplots_adjust(
        bottom=0.22
    )

    png_file = (
        output_prefix.with_suffix(
            ".png"
        )
    )

    pdf_file = (
        output_prefix.with_suffix(
            ".pdf"
        )
    )

    svg_file = (
        output_prefix.with_suffix(
            ".svg"
        )
    )

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

    plt.close(
        figure
    )

    print(
        f"\nSaved plot: {png_file}"
    )

    print(
        f"Saved plot: {pdf_file}"
    )

    print(
        f"Saved plot: {svg_file}"
    )


# ---------------------------------------------------------------------
# COMMAND LINE
# ---------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Compare CLEAN EC predictions against "
            "reference EC labels at hierarchy levels "
            "1, 2, 3, and 4."
        )
    )

    # -------------------------------------------------------------
    # MOST
    # -------------------------------------------------------------

    parser.add_argument(
        "--most-csv",
        type=Path,
        default=None,
        help=(
            "CLEAN prediction file for the "
            ">10,000 training-sequence group."
        ),
    )

    parser.add_argument(
        "--most-ec",
        type=Path,
        default=None,
        help=(
            "Reference EC file for the "
            ">10,000 training-sequence group."
        ),
    )

    # -------------------------------------------------------------
    # MIDDLE
    # -------------------------------------------------------------

    parser.add_argument(
        "--middle-csv",
        type=Path,
        default=None,
        help=(
            "CLEAN prediction file for the "
            "100–10,000 training-sequence group."
        ),
    )

    parser.add_argument(
        "--middle-ec",
        type=Path,
        default=None,
        help=(
            "Reference EC file for the "
            "100–10,000 training-sequence group."
        ),
    )

    # -------------------------------------------------------------
    # LOW
    # -------------------------------------------------------------

    parser.add_argument(
        "--low-csv",
        type=Path,
        default=None,
        help=(
            "CLEAN prediction file for the "
            "<100 training-sequence group."
        ),
    )

    parser.add_argument(
        "--low-ec",
        type=Path,
        default=None,
        help=(
            "Reference EC file for the "
            "<100 training-sequence group."
        ),
    )

    # -------------------------------------------------------------
    # VALIDATION COMMON EC
    # -------------------------------------------------------------

    parser.add_argument(
        "--val-com-csv",
        type=Path,
        default=None,
        help=(
            "CLEAN prediction file for validation "
            "reactions whose EC exists in training."
        ),
    )

    parser.add_argument(
        "--val-com-ec",
        type=Path,
        default=None,
        help=(
            "Reference EC file for validation "
            "reactions whose EC exists in training."
        ),
    )

    # -------------------------------------------------------------
    # VALIDATION DIFFERENT EC
    # -------------------------------------------------------------

    parser.add_argument(
        "--val-dif-csv",
        type=Path,
        default=None,
        help=(
            "CLEAN prediction file for validation "
            "reactions whose EC is absent from training."
        ),
    )

    parser.add_argument(
        "--val-dif-ec",
        type=Path,
        default=None,
        help=(
            "Reference EC file for validation "
            "reactions whose EC is absent from training."
        ),
    )

    # -------------------------------------------------------------
    # GENERAL OPTIONS
    # -------------------------------------------------------------

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "clean_ec_accuracy"
        ),
        help=(
            "Output filename prefix. "
            "Default: clean_ec_accuracy"
        ),
    )

    parser.add_argument(
        "--prediction-mode",
        choices=[
            "any",
            "first",
            "lowest-score",
        ],
        default="any",
        help=(
            "How to evaluate multiple CLEAN EC predictions. "
            "Default: any."
        ),
    )

    parser.add_argument(
        "--aggregation",
        choices=[
            "micro",
            "macro",
        ],
        default="micro",
        help=(
            "micro: total correct sequences divided by "
            "total evaluated sequences. "
            "macro: mean percentage across reactions. "
            "Default: micro."
        ),
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # DETERMINE WHICH GROUPS WERE PROVIDED
    # -------------------------------------------------------------

    active_groups: list[
        dict[str, str]
    ] = []

    group_files: list[
        tuple[
            dict[str, str],
            Path,
            Path,
        ]
    ] = []

    for group in GROUPS:

        prediction_file = getattr(
            args,
            group[
                "csv_argument"
            ],
        )

        ground_truth_file = getattr(
            args,
            group[
                "ec_argument"
            ],
        )

        # Neither was supplied -> simply skip this group.
        if (
            prediction_file is None
            and ground_truth_file is None
        ):
            continue

        # One supplied without the other -> error.
        if prediction_file is None:

            raise SystemExit(
                f"Reference EC file was supplied "
                f"for group '{group['name']}', "
                f"but the CLEAN prediction CSV "
                f"was not supplied."
            )

        if ground_truth_file is None:

            raise SystemExit(
                f"CLEAN prediction CSV was supplied "
                f"for group '{group['name']}', "
                f"but the reference EC file "
                f"was not supplied."
            )

        prediction_file = (
            prediction_file
            .expanduser()
            .resolve()
        )

        ground_truth_file = (
            ground_truth_file
            .expanduser()
            .resolve()
        )

        if not prediction_file.is_file():

            raise SystemExit(
                f"CLEAN prediction file does not exist: "
                f"{prediction_file}"
            )

        if not ground_truth_file.is_file():

            raise SystemExit(
                f"Reference EC file does not exist: "
                f"{ground_truth_file}"
            )

        active_groups.append(
            group
        )

        group_files.append(
            (
                group,
                prediction_file,
                ground_truth_file,
            )
        )

    # At least one pair is required.
    if not group_files:

        parser.error(
            "No input files were supplied. "
            "Provide at least one CLEAN prediction/reference pair, "
            "for example:\n\n"
            "  --most-csv most_merged_output.csv "
            "--most-ec most_smiles_with_ec.tsv"
        )

    # -------------------------------------------------------------
    # OUTPUT PATH
    # -------------------------------------------------------------

    output_prefix = (
        args.output
        .expanduser()
        .resolve()
    )

    output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # RUN ANALYSIS
    # -------------------------------------------------------------

    all_reaction_results: list[
        dict[str, object]
    ] = []

    group_summaries: list[
        dict[str, object]
    ] = []

    print(
        "\nGroups selected for analysis:"
    )

    for (
        group,
        prediction_file,
        ground_truth_file,
    ) in group_files:

        print(
            f"\nGroup: {group['name']}"
        )

        print(
            f"  CLEAN predictions: "
            f"{prediction_file}"
        )

        print(
            f"  Reference ECs:     "
            f"{ground_truth_file}"
        )

        reaction_results = (
            evaluate_group(
                group=group,
                prediction_file=prediction_file,
                ground_truth_file=ground_truth_file,
                prediction_mode=args.prediction_mode,
            )
        )

        summary = (
            summarize_group(
                group=group,
                reaction_results=reaction_results,
                aggregation=args.aggregation,
            )
        )

        all_reaction_results.extend(
            reaction_results
        )

        group_summaries.append(
            summary
        )

    # -------------------------------------------------------------
    # SAVE TABLES
    # -------------------------------------------------------------

    per_reaction_file = (
        output_prefix.parent
        / (
            f"{output_prefix.name}"
            f"_per_reaction.tsv"
        )
    )

    summary_file = (
        output_prefix.parent
        / (
            f"{output_prefix.name}"
            f"_group_summary.tsv"
        )
    )

    write_per_reaction_results(
        all_reaction_results,
        per_reaction_file,
    )

    write_group_summary(
        group_summaries,
        summary_file,
    )

    # -------------------------------------------------------------
    # PRINT AND PLOT
    # -------------------------------------------------------------

    print_summary(
        group_summaries,
        aggregation=args.aggregation,
    )

    make_plot(
        group_summaries,
        active_groups=active_groups,
        output_prefix=output_prefix,
        aggregation=args.aggregation,
    )

    print(
        f"\nSaved per-reaction results: "
        f"{per_reaction_file}"
    )

    print(
        f"Saved group summary:       "
        f"{summary_file}"
    )


if __name__ == "__main__":
    main()
