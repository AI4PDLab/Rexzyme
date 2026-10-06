"""Download REXzyme datasets from the Hugging Face model repo.

The paper checkpoint, tokenizers, and train/validation JSON live in
https://huggingface.co/AI4PD/REXzyme
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_HF_REPO = "AI4PD/REXzyme"
DEFAULT_CHECKPOINT = f"{DEFAULT_HF_REPO}/checkpoint-90000"
DEFAULT_TOKENIZER_AA = f"{DEFAULT_HF_REPO}/tokenizer_aa"
DEFAULT_TOKENIZER_MOL = f"{DEFAULT_HF_REPO}/tokenizer_smiles"
TRAIN_JSON_NAME = "training_dataset.json"
VAL_JSON_NAME = "validation_dataset.json"


def is_hub_repo_id(spec: str) -> bool:
    """True for ``owner/name`` (not a local path or a nested file path)."""
    if not spec or os.path.exists(spec):
        return False
    parts = spec.strip("/").split("/")
    return len(parts) == 2 and "." not in parts[-1]


def resolve_hub_file(
    spec: str,
    *,
    filename: str | None = None,
    repo_id: str = DEFAULT_HF_REPO,
) -> str:
    """Return a local filesystem path for a JSON (or other) file.

    ``spec`` may be:

    * an existing local file
    * a Hub repo id (``AI4PD/REXzyme``) — downloads ``filename``
    * a filename in the default repo (``training_dataset.json``)
    """
    if spec and os.path.isfile(spec):
        return os.path.abspath(spec)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise SystemExit(
            "huggingface_hub is required to download AI4PD/REXzyme files. "
            "Install it with: pip install huggingface_hub"
        ) from error

    if filename:
        download_repo = spec if is_hub_repo_id(spec) else repo_id
        download_name = filename
    elif is_hub_repo_id(spec):
        download_repo = spec
        download_name = TRAIN_JSON_NAME
    else:
        download_repo = repo_id
        download_name = spec

    print(f"Downloading {download_name} from {download_repo} …")
    path = hf_hub_download(repo_id=download_repo, filename=download_name)
    print(f"Cached at {path}")
    return path


def resolve_train_json(spec: str | os.PathLike | None = None) -> Path:
    return Path(resolve_hub_file(str(spec or DEFAULT_HF_REPO), filename=TRAIN_JSON_NAME))


def resolve_val_json(spec: str | os.PathLike | None = None) -> Path:
    return Path(resolve_hub_file(str(spec or DEFAULT_HF_REPO), filename=VAL_JSON_NAME))
