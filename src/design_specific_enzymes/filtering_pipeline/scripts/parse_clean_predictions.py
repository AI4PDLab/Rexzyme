#!/usr/bin/env python3
"""Parse CLEAN prediction output into a standardized metric CSV."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd

PRED_PATTERN = re.compile(r"EC:([^/,\s]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse CLEAN prediction output.")
    parser.add_argument("--clean_output", required=True, help="CLEAN output file.")
    parser.add_argument("--input_fasta", required=True, help="Original FASTA used for CLEAN.")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--target_ec", default="")
    return parser.parse_args()


def read_fasta(path: Path) -> list[str]:
    sequences = []
    current = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
                current = []
        else:
            current.append(line.strip())
    if current:
        sequences.append("".join(current))
    return sequences


def md5(sequence: str) -> str:
    return hashlib.md5(sequence.encode("utf-8")).hexdigest()


def sample_to_index(sample_id: str) -> int | None:
    sample_id = sample_id.strip()
    if "_" in sample_id:
        candidate = sample_id.split("_")[-1]
    else:
        candidate = sample_id
    try:
        return int(candidate)
    except ValueError:
        return None


def first_predicted_ec(text: str) -> str:
    match = PRED_PATTERN.search(text or "")
    return match.group(1) if match else ""


def main() -> None:
    args = parse_args()
    sequences = read_fasta(Path(args.input_fasta))
    rows = []
    with open(args.clean_output, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            fields = line.split(maxsplit=1)
            sample_id = fields[0]
            pred_text = fields[1] if len(fields) > 1 else ""
            seq_index = sample_to_index(sample_id)
            if seq_index is None or seq_index < 0 or seq_index >= len(sequences):
                continue
            sequence = sequences[seq_index]
            rows.append(
                {
                    "sample_id": sample_id,
                    "sequence_index": seq_index,
                    "sequence_md5": md5(sequence),
                    "predicted_ec": first_predicted_ec(pred_text),
                    "target_ec": args.target_ec,
                }
            )

    out = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(out)} parsed CLEAN rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
