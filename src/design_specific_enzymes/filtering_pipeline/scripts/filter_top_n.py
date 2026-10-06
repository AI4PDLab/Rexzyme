#!/usr/bin/env python3
"""Keep top N rows by one numeric score column."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep top N rows by score.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--column", required=True)
    parser.add_argument("--top_n", type=int, required=True)
    parser.add_argument(
        "--direction",
        choices=["higher_is_better", "lower_is_better"],
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("--top_n must be > 0.")

    df = pd.read_csv(args.input_csv)
    if args.column not in df.columns:
        raise ValueError(f"Column not found: {args.column}")

    df = df.copy()
    df["_score"] = pd.to_numeric(df[args.column], errors="coerce")
    df = df[df["_score"].notna()]
    ascending = args.direction == "lower_is_better"
    out = df.sort_values("_score", ascending=ascending).head(args.top_n).drop(columns=["_score"])

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Kept {len(out)}/{len(df)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
