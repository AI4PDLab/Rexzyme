"""Teacher-forced scoring of T5 generate() output must reproduce the generation-time log-probs."""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from src.sequence_scoring import generated_sequence_nll  # noqa: E402

EOS = 1


def tiny_t5():
    torch.manual_seed(0)
    config = transformers.T5Config(
        vocab_size=24, d_model=16, d_ff=32, d_kv=8, num_layers=1, num_heads=2,
        pad_token_id=0, eos_token_id=EOS, decoder_start_token_id=0,
    )
    return transformers.T5ForConditionalGeneration(config).eval()


SOURCE = torch.tensor([[5, 6, 7, 8, EOS]])


def sample(model):
    torch.manual_seed(0)
    with torch.no_grad():
        return model.generate(
            SOURCE, do_sample=True, top_k=0, max_new_tokens=12, num_return_sequences=8,
            return_dict_in_generate=True, output_logits=True,
        )


def generation_nll(out) -> list[float]:
    """Mean -log-prob the model gave its own tokens while generating (raw logits, up to EOS)."""
    log_probs = torch.log_softmax(torch.stack(out.logits, dim=1).float(), dim=-1)
    nlls = []
    for row, row_log_probs in zip(out.sequences[:, 1:], log_probs):
        total, count = 0.0, 0
        for position, token in enumerate(row.tolist()):
            total -= row_log_probs[position, token].item()
            count += 1
            if token == EOS:
                break
        nlls.append(total / count)
    return nlls


def test_scores_match_generation_and_the_old_labels_approach_does_not() -> None:
    model = tiny_t5()
    out = sample(model)
    assert (out.sequences[:, 0] == model.config.decoder_start_token_id).all()
    assert len({int((row == EOS).nonzero()[0]) if (row == EOS).any() else -1 for row in out.sequences}) > 1

    expected = generation_nll(out)
    assert generated_sequence_nll(model, out.sequences, SOURCE) == pytest.approx(expected, abs=1e-5)

    old = []  # before the fix: the whole generate() output, start token included, as labels
    for row in out.sequences:
        labels = row.unsqueeze(0).clone()
        labels[labels == model.config.pad_token_id] = -100
        with torch.no_grad():
            old.append(model(input_ids=SOURCE, labels=labels).loss.item())
    assert old != pytest.approx(expected, abs=1e-3)


def test_padding_after_eos_is_not_scored() -> None:
    model = tiny_t5()
    finished = torch.tensor([[0, 4, 5, EOS]])
    padded = torch.tensor([[0, 4, 5, EOS, 0, 0, 0]])
    assert generated_sequence_nll(model, padded, SOURCE) == pytest.approx(
        generated_sequence_nll(model, finished, SOURCE)
    )
