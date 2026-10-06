#!/usr/bin/env python3
"""Keep rows where predicted EC matches target EC at a selected level."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def ec_prefix(ec: str, level: int) -> str:
    parts = str(ec).replace("EC:", "").strip().split(".")
    return ".".join(parts[:level])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter by CLEAN EC match.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--predicted_ec_column", default="predicted_ec")
    parser.add_argument("--target_ec_column", default="target_ec")
    parser.add_argument("--level", type=int, choices=[1, 2, 3, 4], default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    for col in [args.predicted_ec_column, args.target_ec_column]:
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")

    pred = df[args.predicted_ec_column].astype(str)
    targ = df[args.target_ec_column].astype(str)
    keep = pred.apply(lambda x: ec_prefix(x, args.level)) == targ.apply(lambda x: ec_prefix(x, args.level))
    out = df[keep].copy()

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Kept {len(out)}/{len(df)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
