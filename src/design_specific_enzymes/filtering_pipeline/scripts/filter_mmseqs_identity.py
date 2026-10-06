#!/usr/bin/env python3
"""Remove rows above a maximum identity-to-natural threshold."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter by MMseqs identity.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--identity_column", default="identity_to_natural")
    parser.add_argument(
        "--max_identity",
        type=float,
        default=0.9999,
        help="Rows with identity >= this value are removed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    if args.identity_column not in df.columns:
        raise ValueError(f"Column not found: {args.identity_column}")

    identity = pd.to_numeric(df[args.identity_column], errors="coerce")
    keep = identity < args.max_identity
    out = df[keep].copy()

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Kept {len(out)}/{len(df)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
