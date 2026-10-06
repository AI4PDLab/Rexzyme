#!/usr/bin/env python3
"""Download Rexzyme model-repo files from Hugging Face.

The paper checkpoint, tokenizers, training JSON, and validation JSON are in
https://huggingface.co/AI4PD/REXzyme

By default this caches ``training_dataset.json`` and ``validation_dataset.json``
(large files) via ``huggingface_hub`` and prints the local paths.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hf_assets import (  # noqa: E402
    DEFAULT_HF_REPO,
    TRAIN_JSON_NAME,
    VAL_JSON_NAME,
    resolve_hub_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=DEFAULT_HF_REPO,
        help="Hugging Face repo id",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=[TRAIN_JSON_NAME, VAL_JSON_NAME],
        help="Filenames inside the repo to download",
    )
    args = parser.parse_args()

    for name in args.files:
        path = resolve_hub_file(args.repo, filename=name)
        print(path)


if __name__ == "__main__":
    main()
