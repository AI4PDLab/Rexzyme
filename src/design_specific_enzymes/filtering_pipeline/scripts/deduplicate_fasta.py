#!/usr/bin/env python3
"""Remove duplicate sequences from a FASTA file."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate FASTA by sequence.")
    parser.add_argument("--input_fasta", required=True)
    parser.add_argument("--output_fasta", required=True)
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


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n{sequence}\n")


def main() -> None:
    args = parse_args()
    records = read_fasta(Path(args.input_fasta))
    seen = set()
    unique_records = []
    for header, sequence in records:
        if sequence in seen:
            continue
        seen.add(sequence)
        unique_records.append((header, sequence))

    out_path = Path(args.output_fasta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_fasta(out_path, unique_records)
    print(f"Kept {len(unique_records)}/{len(records)} unique sequences -> {out_path}")


if __name__ == "__main__":
    main()
