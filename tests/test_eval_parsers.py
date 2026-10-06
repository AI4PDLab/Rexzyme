from pathlib import Path

from eval.gt_superpose import apply_reliability_filter as reliability
from eval.gt_superpose import download_gt_pdbs as download_gt
from eval.train_identity import mmseq_file_preparation as mmseq_prep


def test_parse_reaction_number_and_smiles_map(tmp_path: Path) -> None:
    smiles_map = tmp_path / "smiles_map.json"
    smiles_map.write_text(
        '{"output_file_2.fasta": "BB", "output_file_0.fasta": "AA", "ignored": "ZZ"}',
        encoding="utf-8",
    )

    assert download_gt.parse_reaction_number("output_file_1255.fasta") == 1255
    assert download_gt.parse_reaction_number("output_file-1255.fasta") == 1255
    assert download_gt.parse_reaction_number("not_a_fasta") is None
    assert download_gt.load_smiles_map(smiles_map) == [(0, "AA"), (2, "BB")]


def test_load_uniprot_tsv_splits_accessions(tmp_path: Path) -> None:
    table = tmp_path / "rhea_uniprot.tsv"
    table.write_text(
        "Reaction\tRheaID\tEC\tMethod\tHas_sequence\tbest_criterion\tbest_accessions\n"
        "C>>O\t15985\t1.17.1.9\tPDB\tyes\tPDB\tP12345,Q67890\n",
        encoding="utf-8",
    )

    rows = download_gt.load_uniprot_tsv(table)

    assert rows["C>>O"]["rhea"] == "15985"
    assert rows["C>>O"]["best_acc"] == ["P12345", "Q67890"]


def test_json_to_fasta_reads_hub_jsonl_sequence_field(tmp_path: Path) -> None:
    jsonl = tmp_path / "training_dataset.json"
    jsonl.write_text(
        '{"translation": {"Reaction": "C>>O", "Sequence": "MPEPTIDE"}}\n'
        '{"translation": {"Reaction": "A>>B", "Sequence": "AAAA"}}\n',
        encoding="utf-8",
    )
    fasta = tmp_path / "training_dataset.fasta"

    mmseq_prep.json_to_fasta(str(jsonl), str(fasta))

    assert fasta.read_text(encoding="utf-8") == ">0\nMPEPTIDE\n>1\nAAAA\n"


def test_json_to_fasta_accepts_json_array_and_protein_sequence_fallback(
    tmp_path: Path,
) -> None:
    array_file = tmp_path / "validation_dataset.json"
    array_file.write_text(
        '[{"translation": {"Reaction": "C>>O", "Protein sequence": "MPEPTIDE"}},'
        ' {"translation": {"Reaction": "A>>B", "Sequence": "WWWW"}}]',
        encoding="utf-8",
    )
    fasta = tmp_path / "validation_dataset.fasta"

    mmseq_prep.json_to_fasta(str(array_file), str(fasta))

    assert fasta.read_text(encoding="utf-8") == ">0\nMPEPTIDE\n>1\nWWWW\n"


def test_collate_fasta_renumbers_headers(tmp_path: Path) -> None:
    folder = tmp_path / "fastas"
    folder.mkdir()
    (folder / "a.fasta").write_text(">first\nACDE\n", encoding="utf-8")
    (folder / "b.fasta").write_text(">second\nFGHI\n", encoding="utf-8")
    output = tmp_path / "collated.fasta"

    mmseq_prep.collate_fasta(str(folder), str(output))
    text = output.read_text(encoding="utf-8")

    assert ">0\n" in text
    assert ">1\n" in text
    assert "ACDE" in text
    assert "FGHI" in text
    assert (tmp_path / "collated_metadata.pkl").is_file()


def test_reliability_filter_keeps_covered_pairs(tmp_path: Path) -> None:
    unfiltered = tmp_path / "all_tmscores_unfiltered.tsv"
    unfiltered.write_text(
        "\n".join(
            [
                "reaction\tdesign\ttcov\ttlen\tqcov\tttmscore",
                "output_file_0\td0\t0.6\t80\t0.6\t0.4",
                "output_file_0\td1\t0.2\t80\t0.6\t0.9",
                "output_file_0\td2\t0.6\t10\t0.6\t0.9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    filtered = tmp_path / "all_tmscores.tsv"

    kept, total = reliability.filter_file(unfiltered, filtered, 0.5, 50, 0.5)
    lines = filtered.read_text(encoding="utf-8").splitlines()

    assert (kept, total) == (1, 3)
    assert lines[1].startswith("output_file_0\td0")
