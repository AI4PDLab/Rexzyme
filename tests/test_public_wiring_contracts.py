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

    assert "src/tokenize.py" in script
    assert "src/train.py" in script
    assert (ROOT / "src" / "tokenize.py").is_file()
    assert (ROOT / "src" / "train.py").is_file()
