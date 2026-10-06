#!/usr/bin/env python3
"""Keep top fraction by one numeric score column."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep top fraction of rows by a score column."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--column", required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument(
        "--direction",
        choices=["higher_is_better", "lower_is_better"],
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.fraction <= 1.0):
        raise ValueError("--fraction must be in (0, 1].")

    df = pd.read_csv(args.input_csv)
    if args.column not in df.columns:
        raise ValueError(f"Column not found: {args.column}")

    df = df.copy()
    df["_score"] = pd.to_numeric(df[args.column], errors="coerce")
    n_input = len(df)
    df = df[df["_score"].notna()]
    if n_input and df.empty:
        raise ValueError(
            f"Column {args.column!r} has no numeric values in {args.input_csv}; "
            "was the metric computed and merged into this table?"
        )

    ascending = args.direction == "lower_is_better"
    n_keep = max(1, int(len(df) * args.fraction))
    out = df.sort_values("_score", ascending=ascending).head(n_keep).drop(columns=["_score"])

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Kept {len(out)}/{len(df)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
