import pytest

pytest.importorskip("numpy")

from eval.gt_acc import gt_mmseq_interpretation_balanced as balanced
from eval.gt_acc import gt_permutation_mmseq_eval as permutation


def test_balanced_mmseq_parser_counts_empty_hits_as_zero(tmp_path) -> None:
    mmseqs_file = tmp_path / "empty.m8"
    mmseqs_file.write_text("", encoding="utf-8")

    assert balanced.analyze_mmseqs_output(str(mmseqs_file), n_inference_seqs=3) == (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def test_balanced_mmseq_parser_uses_best_hit_per_query(tmp_path) -> None:
    mmseqs_file = tmp_path / "hits.m8"
    mmseqs_file.write_text(
        "\n".join(
            [
                "0\tgt_a\t0.40\t50\t100\t100",
                "0\tgt_b\t0.80\t25\t100\t100",
                "1\tgt_c\t0.25\t40\t100\t80",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    best, mean, std, best_eff, mean_eff, std_eff = balanced.analyze_mmseqs_output(
        str(mmseqs_file),
        n_inference_seqs=3,
    )

    assert best == 0.80
    assert mean == (0.80 + 0.25 + 0.0) / 3
    assert std >= 0.0
    assert best_eff == 0.2
    assert mean_eff == (0.2 + 0.125 + 0.0) / 3
    assert std_eff >= 0.0


def test_permutation_pairwise_parser_keeps_best_effective_identity(tmp_path) -> None:
    mmseqs_file = tmp_path / "pairwise.m8"
    mmseqs_file.write_text(
        "\n".join(
            [
                "0\t0\t0.50\t40\t100\t100",
                "0\t1\t0.40\t100\t100\t100",
                "1\t0\t0.20\t50\t100\t100",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = permutation.parse_pairwise_best_hits(
        str(mmseqs_file),
        query_records=[
            ("output_topk15_file_0.fasta", "query_smiles", "0", 0),
            ("output_topk15_file_0.fasta", "query_smiles", "1", 1),
        ],
        unique_targets=["GT_A", "GT_B"],
        mol_to_gt={"target_smiles": {"GT_A", "GT_B"}},
    )

    assert parsed[("query_smiles", "target_smiles")][0] == {
        "identity": 0.4,
        "eff": 0.4,
    }
    assert parsed[("query_smiles", "target_smiles")][1] == {
        "identity": 0.2,
        "eff": 0.1,
    }
