#!/usr/bin/env python3
"""Filter a CSV table by numeric threshold on one column."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter rows by threshold on a numeric column."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--column", required=True)
    parser.add_argument("--op", choices=["gt", "ge", "lt", "le", "eq"], required=True)
    parser.add_argument("--value", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    if args.column not in df.columns:
        raise ValueError(f"Column not found: {args.column}")

    values = pd.to_numeric(df[args.column], errors="coerce")
    if args.op == "gt":
        keep = values > args.value
    elif args.op == "ge":
        keep = values >= args.value
    elif args.op == "lt":
        keep = values < args.value
    elif args.op == "le":
        keep = values <= args.value
    else:
        keep = values == args.value

    out = df[keep].copy()
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Kept {len(out)}/{len(df)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
