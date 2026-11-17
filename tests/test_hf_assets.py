from pathlib import Path

from src import hf_assets


def test_is_hub_repo_id_accepts_owner_name() -> None:
    assert hf_assets.is_hub_repo_id("AI4PD/REXzyme") is True
    assert hf_assets.is_hub_repo_id("AI4PD/REXzyme/tokenizer_aa") is False
    assert hf_assets.is_hub_repo_id("training_dataset.json") is False
    assert hf_assets.is_hub_repo_id("") is False


def test_resolve_hub_file_returns_existing_local_path(tmp_path: Path) -> None:
    local = tmp_path / "training_dataset.json"
    local.write_text("[]\n", encoding="utf-8")

    resolved = hf_assets.resolve_hub_file(str(local), filename="training_dataset.json")

    assert Path(resolved) == local.resolve()


def test_resolve_train_json_uses_local_file_without_download(tmp_path: Path) -> None:
    local = tmp_path / "training_dataset.json"
    local.write_text('{"translation": {}}\n', encoding="utf-8")

    assert hf_assets.resolve_train_json(local) == local.resolve()


def test_default_asset_ids_match_public_hub_layout() -> None:
    assert hf_assets.DEFAULT_HF_REPO == "AI4PD/REXzyme"
    assert hf_assets.DEFAULT_CHECKPOINT.endswith("/checkpoint-90000")
    assert hf_assets.DEFAULT_TOKENIZER_AA.endswith("/tokenizer_aa")
    assert hf_assets.DEFAULT_TOKENIZER_MOL.endswith("/tokenizer_smiles")
    assert hf_assets.TRAIN_JSON_NAME == "training_dataset.json"
    assert hf_assets.VAL_JSON_NAME == "validation_dataset.json"
