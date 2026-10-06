from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers", exc_type=ImportError)

from src import inference


def test_compute_transition_scores_returns_zero_without_logits() -> None:
    import torch

    sequences = torch.tensor([[0]])

    assert inference.compute_transition_scores(sequences, [], pad_token_id=0) == [0.0]


def test_compute_transition_scores_prefers_the_selected_token() -> None:
    import torch

    sequences = torch.tensor([[0, 1]])
    scores = [torch.tensor([[0.0, 10.0, -10.0]])]

    values = inference.compute_transition_scores(
        sequences,
        scores,
        pad_token_id=0,
        normalize_by_length=True,
    )

    assert len(values) == 1
    assert values[0] < 0.0
    assert values[0] > -1.0


def test_load_reactions_from_smiles_map_jsonl_and_txt(tmp_path: Path) -> None:
    from eval.embedding_distance import extract_encoder_embeddings as embeddings

    smiles_map = tmp_path / "smiles_map.json"
    smiles_map.write_text(
        '{"output_file_0.fasta": "C>>O", "output_file_1.fasta": "C>>O"}\n',
        encoding="utf-8",
    )
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(
        '{"translation": {"Reaction": "A>>B", "Sequence": "M"}}\n',
        encoding="utf-8",
    )
    txt = tmp_path / "reactions.txt"
    txt.write_text("C>>O\n\nN>>N\nC>>O\n", encoding="utf-8")

    assert embeddings.load_reactions(smiles_map) == ["C>>O"]
    assert embeddings.load_reactions(jsonl) == ["A>>B"]
    assert embeddings.load_reactions(txt) == ["C>>O", "N>>N"]
    assert embeddings.describe_embedding_source(0) == (
        "the encoder embeddings (hidden_states[0])"
    )
