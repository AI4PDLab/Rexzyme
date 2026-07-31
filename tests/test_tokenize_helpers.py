from pathlib import Path

from src import tokenize


def test_gather_splits_from_directory_uses_expected_split_names(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text('{"translation": {}}\n', encoding="utf-8")
    (tmp_path / "val.jsonl").write_text('{"translation": {}}\n', encoding="utf-8")
    (tmp_path / "test.json").write_text('[{"translation": {}}]\n', encoding="utf-8")

    assert tokenize.gather_splits_from_directory(str(tmp_path)) == {
        "train": str(tmp_path / "train.jsonl"),
        "validation": str(tmp_path / "val.jsonl"),
        "test": str(tmp_path / "test.json"),
    }


def test_resolve_local_data_files_prefers_explicit_split_mapping(tmp_path: Path) -> None:
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    explicit = data_dir / "custom-val.jsonl"
    explicit.write_text('{"translation": {}}\n', encoding="utf-8")

    assert tokenize.resolve_local_data_files(
        str(data_dir),
        {"validation": "custom-val.jsonl"},
    ) == {"validation": str(explicit)}


def test_detect_source_target_for_reaction_sequence_schema() -> None:
    assert tokenize.detect_source_target(
        {"Reaction": "A>>B", "Sequence": "MPEPTIDE"}
    ) == ("Reaction", "Sequence")


def test_detect_source_target_for_smiles_protein_sequence_schema() -> None:
    assert tokenize.detect_source_target(
        {"Canonical Smiles": "CCO", "Protein sequence": "MPEPTIDE"}
    ) == ("Canonical Smiles", "Protein sequence")
