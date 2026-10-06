from pathlib import Path

import pytest

pytest.importorskip("matplotlib", exc_type=ImportError)
pytest.importorskip("numpy")

from eval.clean import plot_clean_ec_accuracy as clean
from eval.plddt import plot_plddt_by_group as plddt


def test_split_ec_labels_ignores_not_found_and_strips_prefix() -> None:
    assert clean.split_ec_labels("Not Found") == []
    assert clean.split_ec_labels("EC:1.17.1.9; 3.2.1.1") == ["1.17.1.9", "3.2.1.1"]


def test_ec_matches_at_level_uses_hierarchy() -> None:
    assert clean.ec_parts("EC:3.2.2.n1") == ("3", "2", "2", "n1")
    assert clean.ec_matches_at_level("3.2.2.n1", "3.2.2.4", 3) is True
    assert clean.ec_matches_at_level("3.2.2.n1", "3.2.2.4", 4) is False
    assert clean.sequence_is_correct(["3.2.2.n1"], ["3.2.2.4"], 3) is True
    assert clean.sequence_is_correct(["3.2.2.n1"], ["3.2.2.4"], 4) is False


def test_convert_plddt_to_percent_handles_both_scales() -> None:
    assert plddt.convert_plddt_to_percent(0.8) == 80.0
    assert plddt.convert_plddt_to_percent(80.0) == 80.0


def test_mean_plddt_from_pdb_uses_ca_bfactors(tmp_path: Path) -> None:
    pdb_path = tmp_path / "design.pdb"
    pdb_path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00\n"
        "ATOM      2  CB  ALA A   1       0.000   0.000   0.000  1.00 10.00\n"
        "ATOM      3  CA  SER A   2       0.000   0.000   0.000  1.00 70.00\n",
        encoding="utf-8",
    )

    assert plddt.mean_plddt_from_pdb(pdb_path) == 80.0


def test_find_pdb_files_requires_fold_directory(tmp_path: Path) -> None:
    group = tmp_path / "generation_most"
    folded = group / "generation_output" / "inference_0" / "fold" / "folder_0"
    folded.mkdir(parents=True)
    (folded / "a.pdb").write_text("ATOM\n", encoding="utf-8")
    skipped = group / "other"
    skipped.mkdir()
    (skipped / "b.pdb").write_text("ATOM\n", encoding="utf-8")

    found = plddt.find_pdb_files(group)

    assert found == [folded / "a.pdb"]
