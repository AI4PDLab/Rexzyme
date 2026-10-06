from pathlib import Path

import pytest

pytest.importorskip("rdkit")

from eval.fp_distance import fingerprint_dataset_distance as fp_distance


def test_load_heldout_and_training_reactions(tmp_path: Path) -> None:
    smiles_map = tmp_path / "smiles_map.json"
    smiles_map.write_text(
        '{"output_file_0.fasta": "C>>O",'
        ' "folder/output_file_1.fasta": "C>>O",'
        ' "output_file_1255.fasta": "N>>N"}',
        encoding="utf-8",
    )
    train_jsonl = tmp_path / "training_dataset.json"
    train_jsonl.write_text(
        '{"translation": {"Reaction": "C>>O", "Sequence": "M"}}\n'
        '{"translation": {"Reaction": "A>>B", "Sequence": "A"}}\n',
        encoding="utf-8",
    )
    train_array = tmp_path / "training_array.json"
    train_array.write_text(
        '[{"translation": {"Reaction": "X>>Y", "Sequence": "X"}}]',
        encoding="utf-8",
    )

    heldout = fp_distance.load_heldout_reactions(smiles_map)
    jsonl_reactions = fp_distance.load_training_reactions(train_jsonl)
    array_reactions = fp_distance.load_training_reactions(train_array)

    assert heldout["C>>O"] == ["output_file_0.fasta", "output_file_1.fasta"]
    assert "output_file_1255.fasta" in heldout["N>>N"]
    assert jsonl_reactions == {"C>>O", "A>>B"}
    assert array_reactions == {"X>>Y"}


def test_identity_plot_parser_scales_percent_identity(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("numpy")
    from eval.train_identity import plot_mmseqs_identity_vs_alignment as identity_plot

    mmseqs = tmp_path / "hits.m8"
    mmseqs.write_text(
        "q1\tt1\t67.7\t40\n"
        "q2\tt2\t0.50\t20\n"
        "bad\n",
        encoding="utf-8",
    )

    lengths, identities = identity_plot.read_mmseqs_file(mmseqs)

    assert list(lengths) == [40.0, 20.0]
    assert identities[0] == 0.677
    assert identities[1] == 0.50
