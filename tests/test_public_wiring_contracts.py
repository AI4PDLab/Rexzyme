import re
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]


class PublicWiringContracts(TestCase):
    def test_inference_input_order_is_stable_for_txt_json_and_jsonl(self) -> None:
        source = (ROOT / "src" / "inference.py").read_text(encoding="utf-8")

        self.assertNotIn("smiles_list = set()", source)
        self.assertRegex(source, r"smiles_list\s*=\s*\[\]")
        self.assertRegex(source, r"\.suffix")
        self.assertRegex(source, r"\.jsonl?")
        self.assertRegex(source, r"GT")

    def test_eval_accepts_inference_metadata_fallback(self) -> None:
        inference_source = (ROOT / "src" / "inference.py").read_text(encoding="utf-8")
        gt_eval_source = (ROOT / "eval" / "gt_acc" / "gt_inference_mmseq.py").read_text(
            encoding="utf-8"
        )
        permutation_source = (
            ROOT / "eval" / "gt_acc" / "gt_permutation_mmseq_eval.py"
        ).read_text(encoding="utf-8")

        self.assertIn("inference_metadata.json", inference_source)
        self.assertRegex(gt_eval_source, re.escape("inference_metadata.json"))
        self.assertRegex(permutation_source, re.escape("inference_metadata.json"))


def test_training_wrapper_points_to_existing_public_entrypoints() -> None:
    script = (ROOT / "scripts" / "run_train.sh").read_text(encoding="utf-8")

    assert "python -m src.tokenize" in script
    assert "python -m src.train" in script
    assert 'DATASET="${DATASET:-AI4PD/REXzyme}"' in script
    assert (ROOT / "src" / "tokenize.py").is_file()
    assert (ROOT / "src" / "train.py").is_file()


def test_inference_wrapper_uses_paper_hub_defaults() -> None:
    script = (ROOT / "scripts" / "run_inference.sh").read_text(encoding="utf-8")

    assert "AI4PD/REXzyme/checkpoint-90000" in script
    assert "AI4PD/REXzyme/tokenizer_aa" in script
    assert "AI4PD/REXzyme/tokenizer_smiles" in script
    assert "python -m src.inference" in script
    assert "TOP_K=${TOP_K:-100}" in script
    assert (ROOT / "src" / "inference.py").is_file()


def test_eval_wrappers_source_hub_and_heldout_range() -> None:
    fp_script = (ROOT / "eval" / "fp_distance" / "run_fp_distance.sh").read_text(
        encoding="utf-8"
    )
    mmseqs_script = (
        ROOT / "eval" / "designs_vs_train" / "run_mmseqs_designs_vs_train.sh"
    ).read_text(encoding="utf-8")
    superpose_script = (
        ROOT / "eval" / "gt_superpose" / "superpose_designs_to_gt.sh"
    ).read_text(encoding="utf-8")
    download_script = (ROOT / "scripts" / "download_hf_assets.py").read_text(
        encoding="utf-8"
    )

    assert "TRAIN_JSON=${TRAIN_JSON:-AI4PD/REXzyme}" in fp_script
    assert "--output" in fp_script
    assert "fingerprint_dataset_distance.py" in fp_script
    assert "all_sequences_with_filenames.fasta" in mmseqs_script
    assert "--query" in mmseqs_script
    assert "--target" in mmseqs_script
    assert "AI4PD/REXzyme" in mmseqs_script
    assert "--designs" in superpose_script
    assert "--end" in superpose_script
    assert "1255" in superpose_script
    assert "training_dataset.json" in download_script
    assert "validation_dataset.json" in download_script
    assert (ROOT / "eval" / "fp_distance" / "fingerprint_dataset_distance.py").is_file()


def test_inference_tokenizer_subfolders_match_hub_layout() -> None:
    source = (ROOT / "src" / "inference.py").read_text(encoding="utf-8")

    assert 'default="tokenizer_aa"' in source
    assert 'default="tokenizer_smiles"' in source
    assert "split_hf_subfolder_ref" in source


def test_tokenize_defaults_to_model_repo_json() -> None:
    source = (ROOT / "src" / "tokenize.py").read_text(encoding="utf-8")

    assert 'DEFAULT_HUB_DATASET = "AI4PD/REXzyme"' in source
    assert "training_dataset.json" in source
    assert "validation_dataset.json" in source


def test_heldout_reaction_list_covers_0_through_1255() -> None:
    reactions = (ROOT / "data" / "generation_heldout.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    smiles_map = (ROOT / "data" / "generated" / "generation_heldout" / "smiles_map.json")

    assert len(reactions) == 1256
    assert smiles_map.is_file()


def test_paper_tmscore_tables_include_reaction_1255() -> None:
    plots = ROOT / "eval" / "tmscore_plots"
    expected = {
        "tmscore_vs_plddt.tsv": "reaction\tdesign\tmean_plddt\tbest_ttmscore",
        "tmscore_vs_fp_distance.tsv": "reaction\tbest_design\tmean_plddt\tmin_distance\tbest_ttmscore",
        "tmscore_vs_seq_id.tsv": "reaction\tbest_design\tmean_plddt\tseq_identity_pct\tbest_ttmscore",
        "tmscore_vs_embed_distance.tsv": "reaction\tbest_design\tmean_plddt\tmin_emb_distance\tbest_ttmscore",
        "embed_distance_vs_fp_distance.tsv": "reaction\tmin_distance\tmin_emb_distance",
    }

    for name, header in expected.items():
        path = plots / name
        text = path.read_text(encoding="utf-8")
        assert text.splitlines()[0] == header
        assert "output_file_1255" in text
