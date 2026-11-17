#!/usr/bin/env python3
"""Apply Rosetta metric gates used in the final filtering stage."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter with Rosetta gates.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)

    parser.add_argument("--rosetta_score_column", default="rosetta_score")
    parser.add_argument("--relative_hydrophobic_sasa_column", default="relative_hydrophobic_sasa")
    parser.add_argument("--net_charge_column", default="net_charge")
    parser.add_argument("--average_sap_column", default="average_sap")
    parser.add_argument("--hydrophobic_ratio_column", default="hydrophobic_sasa_to_ideal_ratio")

    parser.add_argument("--relative_hydrophobic_sasa_max", type=float, default=3.0)
    parser.add_argument("--average_sap_max", type=float, default=0.5)
    parser.add_argument("--hydrophobic_ratio_max", type=float, default=1.7)
    parser.add_argument("--require_nonzero_net_charge", action="store_true")
    parser.add_argument("--top_n_by_rosetta", type=int, default=0, help="0 means keep all that pass gates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    required = [
        args.rosetta_score_column,
        args.relative_hydrophobic_sasa_column,
        args.net_charge_column,
        args.average_sap_column,
        args.hydrophobic_ratio_column,
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")

    out = df.copy()
    out["_rosetta"] = pd.to_numeric(out[args.rosetta_score_column], errors="coerce")
    out["_rhsasa"] = pd.to_numeric(out[args.relative_hydrophobic_sasa_column], errors="coerce")
    out["_charge"] = pd.to_numeric(out[args.net_charge_column], errors="coerce")
    out["_sap"] = pd.to_numeric(out[args.average_sap_column], errors="coerce")
    out["_hratio"] = pd.to_numeric(out[args.hydrophobic_ratio_column], errors="coerce")

    keep = (
        out["_rhsasa"].le(args.relative_hydrophobic_sasa_max)
        & out["_sap"].le(args.average_sap_max)
        & out["_hratio"].le(args.hydrophobic_ratio_max)
    )
    if args.require_nonzero_net_charge:
        keep &= out["_charge"].ne(0)

    out = out[keep].copy()
    out = out[out["_rosetta"].notna()].sort_values("_rosetta", ascending=True)
    if args.top_n_by_rosetta > 0:
        out = out.head(args.top_n_by_rosetta)

    out = out.drop(columns=["_rosetta", "_rhsasa", "_charge", "_sap", "_hratio"])
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Kept {len(out)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
