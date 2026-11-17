"""Teacher-forced scoring of generate() output must reproduce the generation-time log-probs.

Tiny random models cover the three ways the decoder prefix can look:
T5 (decoder start == pad), BART (decoder start == EOS != pad) and GPT-2 (decoder-only,
prompt prefix, pad == EOS).
"""

from __future__ import annotations

import math

import pytest
import torch

transformers = pytest.importorskip("transformers")

from src.sequence_scoring import (  # noqa: E402
    generated_token_mask,
    sequence_log_likelihood,
    sequence_perplexities,
)

VOCAB = 24


def tiny_t5():
    config = transformers.T5Config(
        vocab_size=VOCAB, d_model=16, d_ff=32, d_kv=8, num_layers=1, num_heads=2,
        pad_token_id=0, eos_token_id=1, decoder_start_token_id=0,
    )
    return transformers.T5ForConditionalGeneration(config).eval()


def tiny_bart():
    config = transformers.BartConfig(
        vocab_size=VOCAB, d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
        max_position_embeddings=64, pad_token_id=1, bos_token_id=0, eos_token_id=2,
        decoder_start_token_id=2, forced_bos_token_id=None, forced_eos_token_id=None,
    )
    return transformers.BartForConditionalGeneration(config).eval()


def tiny_gpt2():
    config = transformers.GPT2Config(
        vocab_size=VOCAB, n_positions=64, n_embd=16, n_layer=1, n_head=2,
        bos_token_id=0, eos_token_id=1, pad_token_id=1,
    )
    return transformers.GPT2LMHeadModel(config).eval()


def generation_log_likelihood(out, prefix_length: int, eos_id: int) -> list[float]:
    """Sum of log-probs the model assigned to its own tokens while generating (until EOS)."""
    log_probs = torch.log_softmax(torch.stack(out.logits, dim=1).float(), dim=-1)  # raw logits
    totals = []
    for row, row_log_probs in zip(out.sequences[:, prefix_length:], log_probs):
        total = 0.0
        for position, token in enumerate(row.tolist()):
            total += row_log_probs[position, token].item()
            if token == eos_id:
                break
        totals.append(total)
    return totals


def sample(model, n_rows: int = 8, **inputs):
    torch.manual_seed(0)
    with torch.no_grad():
        return model.generate(
            **inputs, do_sample=True, top_k=0, max_new_tokens=12, num_return_sequences=n_rows,
            return_dict_in_generate=True, output_logits=True,
        )


def old_labels_log_likelihood(model, input_ids, sequences, pad_id):
    """What the scripts did before: pass generate() output back as `labels`."""
    totals = []
    for row in sequences:
        labels = row.unsqueeze(0).clone()
        labels[labels == pad_id] = -100
        with torch.no_grad():
            loss = model(input_ids=input_ids, labels=labels).loss.item()
        totals.append(-loss * int((labels != -100).sum()))
    return totals


@pytest.mark.parametrize("build", [tiny_t5, tiny_bart], ids=["t5_start_is_pad", "bart_start_is_eos"])
def test_encoder_decoder_scores_match_generation(build) -> None:
    torch.manual_seed(0)
    model = build()
    source = torch.tensor([[5, 6, 7, 8, model.config.eos_token_id]])
    out = sample(model, input_ids=source)
    assert out.sequences[0, 0].item() == model.config.decoder_start_token_id

    expected = generation_log_likelihood(out, prefix_length=1, eos_id=model.config.eos_token_id)
    with torch.no_grad():
        scored, _, _ = sequence_log_likelihood(model, out.sequences, model_inputs={"input_ids": source})
    assert scored.tolist() == pytest.approx(expected, abs=1e-4)

    old = old_labels_log_likelihood(model, source, out.sequences, model.config.pad_token_id)
    assert old != pytest.approx(expected, abs=1e-3)  # the double-start-token bug


def test_decoder_only_scores_match_generation() -> None:
    torch.manual_seed(0)
    model = tiny_gpt2()
    prompt = torch.tensor([[3, 4, 5, 6]])
    out = sample(model, input_ids=prompt, attention_mask=torch.ones_like(prompt))

    expected = generation_log_likelihood(out, prefix_length=prompt.size(1), eos_id=1)
    with torch.no_grad():
        scored, _, _ = sequence_log_likelihood(
            model, out.sequences, prefix_length=prompt.size(1), attention_mask=torch.ones_like(prompt)
        )
    assert scored.tolist() == pytest.approx(expected, abs=1e-4)


def test_rows_are_scored_independently_of_the_batch() -> None:
    torch.manual_seed(0)
    model = tiny_t5()
    source = torch.tensor([[5, 6, 7, 1]])
    out = sample(model, input_ids=source)
    batched = sequence_perplexities(model, out.sequences, model_inputs={"input_ids": source})
    single = [
        sequence_perplexities(model, row.unsqueeze(0), model_inputs={"input_ids": source})[0]
        for row in out.sequences
    ]
    assert batched == pytest.approx(single, rel=1e-5)
    chunked = sequence_perplexities(model, out.sequences, model_inputs={"input_ids": source}, batch_size=3)
    assert chunked == pytest.approx(single, rel=1e-5)


def test_mask_keeps_tokens_up_to_the_first_eos_even_when_pad_is_eos() -> None:
    targets = torch.tensor([[4, 5, 1, 1, 1], [4, 5, 6, 7, 8]])
    assert generated_token_mask(targets, eos_token_id=1, pad_token_id=1).tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]
    # T5-style: pad (0) != eos (1); padding after EOS is dropped, several EOS ids work, and a
    # pad id generated *before* the EOS is still part of the sequence
    targets = torch.tensor([[4, 9, 0, 0], [4, 0, 1, 0]])
    assert generated_token_mask(targets, eos_token_id=[1, 9], pad_token_id=0).tolist() == [
        [True, True, False, False],
        [True, True, True, False],
    ]
    # without EOS ids, pad tokens mark the padding
    assert generated_token_mask(torch.tensor([[4, 5, 0, 0]]), pad_token_id=0).tolist() == [
        [True, True, False, False]
    ]


def test_decoder_only_requires_the_prompt_length() -> None:
    with pytest.raises(ValueError, match="prefix_length"):
        sequence_log_likelihood(tiny_gpt2(), torch.tensor([[3, 4, 5]]))


def test_nothing_generated_gives_nan_perplexity() -> None:
    model = tiny_t5()
    ppl = sequence_perplexities(model, torch.tensor([[0]]), model_inputs={"input_ids": torch.tensor([[5, 1]])})
    assert math.isnan(ppl[0])
