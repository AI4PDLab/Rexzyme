"""Run CLEAN inference on one FASTA placed in CLEAN/app/data/inputs.

This is a thin wrapper around CLEAN's inference utilities, kept in the Rexzyme
repository to document and reproduce the EC-prediction workflow used for
generated sequences.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from CLEAN.infer import infer_maxsep
from CLEAN.utils import prepare_infer_fasta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CLEAN inference for one FASTA file. "
            "The FASTA must exist at CLEAN/app/data/inputs/<name>.fasta "
            "or be passed with --input-fasta."
        )
    )
    parser.add_argument(
        "-d",
        "--fasta_data",
        default=None,
        help=(
            "Input FASTA basename or relative path under data/. "
            "Example: output_topk100_file_1 or "
            "inputs/output_topk100_file_1"
        ),
    )
    parser.add_argument(
        "--input-fasta",
        "--input",
        dest="input_fasta",
        default=None,
        help="Path to a FASTA file. Copied into CLEAN/app/data/inputs/ before inference.",
    )
    return parser.parse_args()


def normalize_test_data(fasta_data: str) -> str:
    fasta_data = fasta_data.strip()
    if fasta_data.endswith(".fasta"):
        fasta_data = fasta_data[:-6]
    if fasta_data.startswith("inputs/"):
        return fasta_data
    return f"inputs/{fasta_data}"


def stage_input_fasta(fasta_path: Path) -> str:
    dest_dir = Path("data") / "inputs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / fasta_path.name
    if dest.resolve() != fasta_path.resolve():
        shutil.copy2(fasta_path, dest)
    return f"inputs/{dest.stem}"


def main() -> None:
    args = parse_args()
    if args.input_fasta:
        fasta_path = Path(args.input_fasta).expanduser().resolve()
        if not fasta_path.is_file():
            raise SystemExit(f"FASTA not found: {fasta_path}")
        test_data = stage_input_fasta(fasta_path)
    elif args.fasta_data:
        test_data = normalize_test_data(args.fasta_data)
    else:
        raise SystemExit("Provide --input-fasta PATH or -d NAME")

    train_data = "split100"

    # Converts the FASTA to the temporary CSV required by CLEAN.
    prepare_infer_fasta(test_data)

    # Writes predictions to results/<test_data>.csv
    infer_maxsep(
        train_data=train_data,
        test_data=test_data,
        report_metrics=False,
        pretrained=True,
        gmm="./data/pretrained/gmm_ensumble.pkl",
    )

    temporary_csv = f"data/{test_data}.csv"
    if os.path.exists(temporary_csv):
        os.remove(temporary_csv)


if __name__ == "__main__":
    main()
