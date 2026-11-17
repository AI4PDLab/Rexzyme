#!/usr/bin/env python3
"""Merge one metric table into the candidate table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge metric outputs into the candidate CSV."
    )
    parser.add_argument("--base_csv", required=True, help="Current candidate table.")
    parser.add_argument("--metric_csv", required=True, help="Metric output table to merge.")
    parser.add_argument("--output_csv", required=True, help="Merged output table.")
    parser.add_argument("--base_key", default="sequence_md5")
    parser.add_argument("--metric_key", default="sequence_md5")
    parser.add_argument(
        "--columns",
        nargs="+",
        required=True,
        help="Metric columns to import from metric_csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.base_csv)
    metric = pd.read_csv(args.metric_csv)

    if args.base_key not in base.columns:
        raise ValueError(f"base key not found: {args.base_key}")
    if args.metric_key not in metric.columns:
        raise ValueError(f"metric key not found: {args.metric_key}")

    missing_cols = [c for c in args.columns if c not in metric.columns]
    if missing_cols:
        raise ValueError(f"Metric columns not found: {missing_cols}")

    use_cols = [args.metric_key] + args.columns
    metric_small = metric[use_cols].copy()
    merged = base.merge(
        metric_small,
        left_on=args.base_key,
        right_on=args.metric_key,
        how="left",
    )
    if args.metric_key != args.base_key:
        merged = merged.drop(columns=[args.metric_key])

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)
    print(
        f"Merged {len(metric_small)} metric rows into {len(base)} candidates -> {args.output_csv}"
    )


if __name__ == "__main__":
    main()
