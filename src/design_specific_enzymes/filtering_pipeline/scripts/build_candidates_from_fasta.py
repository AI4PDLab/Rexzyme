#!/usr/bin/env python3
"""Build an initial candidate CSV from seq_gen-style FASTA output."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

HEADER_PPL = re.compile(r"ppl[_=]([0-9]*\.?[0-9]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create candidates table from generated FASTA."
    )
    parser.add_argument("--input_fasta", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--target_ec", default="", help="Optional fixed target EC.")
    parser.add_argument("--reaction", default="", help="Optional fixed reaction SMILES.")
    return parser.parse_args()


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = None
    seq_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_lines)))
            header = line[1:].strip()
            seq_lines = []
        else:
            seq_lines.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq_lines)))
    return records


def sequence_md5(seq: str) -> str:
    return hashlib.md5(seq.encode("utf-8")).hexdigest()


def extract_ppl(header: str) -> float | None:
    match = HEADER_PPL.search(header)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    records = read_fasta(Path(args.input_fasta))

    rows = []
    for idx, (header, sequence) in enumerate(records):
        rows.append(
            {
                "candidate_id": idx,
                "header": header,
                "sequence": sequence,
                "sequence_md5": sequence_md5(sequence),
                "length": len(sequence),
                "perplexity": extract_ppl(header),
                "target_ec": args.target_ec,
                "reaction": args.reaction,
            }
        )

    out = pd.DataFrame(rows)
    if len(out) and out["perplexity"].isna().all():
        print(
            "WARNING: no 'ppl_<value>' found in any FASTA header, so the perplexity "
            "column is empty and the perplexity filter will fail. Use the FASTA written "
            "by dpo_optimize_and_generate.py, or add perplexity values to the table "
            "(e.g. with merge_metric_table.py).",
            file=sys.stderr,
        )
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(out)} candidates -> {args.output_csv}")


if __name__ == "__main__":
    main()
