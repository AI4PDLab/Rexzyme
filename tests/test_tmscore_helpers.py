from pathlib import Path

import pytest

pytest.importorskip("matplotlib", exc_type=ImportError)
pytest.importorskip("numpy")
pytest.importorskip("scipy")

from eval.tmscore_plots import common
from eval.tmscore_plots import plot_tmscore_vs_embed_distance as embed_plot
from eval.tmscore_plots import plot_tmscore_vs_fp_distance as fp_plot
from eval.tmscore_plots import plot_tmscore_vs_seq_id as seq_plot


def test_reaction_number_parses_underscore_and_hyphen_names() -> None:
    assert common.reaction_number("output_file_1255") == 1255
    assert common.reaction_number("output_file-12") == 12
    assert common.reaction_number("not_a_reaction") is None


def test_design_stem_strips_structure_suffixes() -> None:
    assert common.design_stem("00022_22_ppl_2.31.pdb") == "00022_22_ppl_2.31"
    assert common.design_stem("/tmp/00001_1_ppl_16.19.cif") == "00001_1_ppl_16.19"


def test_read_best_tmscores_keeps_highest_score_per_design(tmp_path: Path) -> None:
    master = tmp_path / "all_tmscores.tsv"
    master.write_text(
        "\n".join(
            [
                "reaction\tdesign\tttmscore",
                "output_file_0\t00001_1_ppl_1.pdb\t0.20",
                "output_file_0\t00001_1_ppl_1.pdb\t0.40",
                "output_file_0\t00002_2_ppl_1.pdb\t0.30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    best = common.read_best_tmscores(master, "ttmscore")
    retained = common.retain_best_design_per_reaction(best)

    assert best[("output_file_0", "00001_1_ppl_1")] == 0.40
    assert list(retained) == [("output_file_0", "00001_1_ppl_1")]
    assert retained[("output_file_0", "00001_1_ppl_1")] == 0.40


def test_read_best_designs_after_plddt_filter_keeps_best_passing_design(
    tmp_path: Path,
) -> None:
    table = tmp_path / "tmscore_vs_plddt.tsv"
    table.write_text(
        "\n".join(
            [
                "reaction\tdesign\tmean_plddt\tbest_ttmscore",
                "output_file_0\t00001_1_ppl_1\t50.0\t0.90",
                "output_file_0\t00002_2_ppl_1\t80.0\t0.40",
                "output_file_0\t00003_3_ppl_1\t90.0\t0.30",
                "output_file_1\t00001_1_ppl_1\t75.0\t0.20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    selected = common.read_best_designs_after_plddt_filter(table, "ttmscore", 70.0)

    assert selected["output_file_0"]["design"] == "00002_2_ppl_1"
    assert selected["output_file_0"]["tm_score"] == 0.40
    assert selected["output_file_1"]["tm_score"] == 0.20


def test_mean_plddt_and_resolve_design_pdb(tmp_path: Path) -> None:
    designs = tmp_path / "output_file-0" / "pdbs"
    designs.mkdir(parents=True)
    pdb_path = designs / "00001_1_ppl_1.pdb"
    pdb_path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 80.00\n"
        "ATOM      2  N   ALA A   1       0.000   0.000   0.000  1.00 10.00\n"
        "ATOM      3  CA  GLY A   2       0.000   0.000   0.000  1.00 60.00\n",
        encoding="utf-8",
    )

    resolved = common.resolve_design_pdb(tmp_path, "output_file_0", "00001_1_ppl_1.pdb")

    assert resolved == pdb_path
    assert common.mean_plddt(pdb_path) == 70.0


def test_fingerprint_and_embedding_parsers_use_output_file_numbers(tmp_path: Path) -> None:
    fp_json = tmp_path / "fingerprint_distances.json"
    fp_json.write_text(
        '[{"fasta_path": "output_file_1255.fasta", "min_distance": 0.25},'
        ' {"fasta_path": "bad", "min_distance": 0.9}]\n',
        encoding="utf-8",
    )
    emb_tsv = tmp_path / "embedding_distance_by_file.tsv"
    emb_tsv.write_text(
        "file_name\tmin_emb_distance\tsmiles\n"
        "output_file_0.fasta\t0.10\tC>>C\n"
        "output_file_1255.fasta\t0.40\tO>>O\n",
        encoding="utf-8",
    )

    assert fp_plot.read_fingerprint_distances(fp_json) == {1255: 0.25}
    assert embed_plot.read_embedding_distances(emb_tsv)[1255] == 0.40


def test_sequence_identity_parser_matches_design_index(tmp_path: Path) -> None:
    mmseqs = tmp_path / "designs_vs_train.best.tsv"
    mmseqs.write_text(
        "design\tpident\n"
        "output_file_0_22_ppl_2.31\t65.7\n"
        "not_a_design\t10.0\n",
        encoding="utf-8",
    )

    assert seq_plot.parse_selected_design_index("00022_22_ppl_2.31") == "22"
    assert seq_plot.parse_mmseqs_design("output_file_0_22_ppl_2.31") == ("0", "22")
    assert seq_plot.parse_mmseqs_design("output_file-0_22_ppl=2.31") == ("0", "22")
    assert seq_plot.read_sequence_identities(mmseqs)[("0", "22")] == 65.7


def test_merge_reaction_data_skips_missing_embedding() -> None:
    selected = {
        "output_file_0": {"design": "d0", "plddt": 80.0, "tm_score": 0.5},
        "output_file-1": {"design": "d1", "plddt": 81.0, "tm_score": 0.4},
        "output_file_2": {"design": "d2", "plddt": 82.0, "tm_score": 0.3},
    }

    xs, ys, rows, missing = embed_plot.merge_reaction_data(selected, {0: 0.2, 1: 0.1})

    assert missing == 1
    assert list(xs) == [0.2, 0.1]
    assert list(ys) == [0.5, 0.4]
    assert rows[0][0] == "output_file_0"
    assert rows[1][0] == "output_file-1"
