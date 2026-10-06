"""compute_transition_scores must ignore the padding generate() appends after EOS."""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers", exc_type=ImportError)

import torch.nn.functional as F

from src import inference

PAD, EOS = 0, 1


def _scores(logits_per_step):
    """List of (batch, vocab) tensors, as returned by generate(output_scores=True)."""
    return [torch.tensor(step, dtype=torch.float32) for step in logits_per_step]


def _logprob(step_logits, batch_index, token):
    return F.log_softmax(torch.tensor(step_logits[batch_index], dtype=torch.float32), dim=-1)[token].item()


# vocab = {0: pad/start, 1: EOS, 2, 3}. Sequence 0 stops after one residue (token 2, then EOS)
# and is padded; sequence 1 runs for three residues plus EOS.
SEQUENCES = torch.tensor([[PAD, 2, EOS, PAD, PAD], [PAD, 3, 2, 3, EOS]])
STEPS = [
    [[-10.0, 0.0, 2.0, 0.0], [-10.0, 0.0, 0.0, 2.0]],   # step 0: tokens 2 / 3
    [[-10.0, 1.0, 0.0, 0.0], [-10.0, 0.0, 1.5, 0.0]],   # step 1: EOS / 2
    [[3.0, 0.0, 0.0, 0.0], [-10.0, 0.0, 0.0, 1.0]],     # step 2: pad (padding of seq 0) / 3
    [[3.0, 0.0, 0.0, 0.0], [-10.0, 2.0, 0.0, 0.0]],     # step 3: pad (padding of seq 0) / EOS
]


def test_positions_after_eos_are_ignored_when_eos_is_given() -> None:
    values = inference.compute_transition_scores(
        SEQUENCES, _scores(STEPS), pad_token_id=PAD, normalize_by_length=True, eos_token_id=EOS
    )
    expected_short = (_logprob(STEPS[0], 0, 2) + _logprob(STEPS[1], 0, EOS)) / 2
    expected_long = sum(
        _logprob(STEPS[i], 1, token) for i, token in enumerate([3, 2, 3, EOS])
    ) / 4
    assert values[0] == pytest.approx(expected_short, abs=1e-6)
    assert values[1] == pytest.approx(expected_long, abs=1e-6)


def test_without_eos_every_position_is_scored_as_before() -> None:
    values = inference.compute_transition_scores(
        SEQUENCES, _scores(STEPS), pad_token_id=PAD, normalize_by_length=True
    )
    all_four = sum(_logprob(STEPS[i], 0, token) for i, token in enumerate([2, EOS, PAD, PAD])) / 4
    assert values[0] == pytest.approx(all_four, abs=1e-6)


def test_score_does_not_depend_on_the_other_sequences_in_the_batch() -> None:
    alone = inference.compute_transition_scores(
        SEQUENCES[:1, :3], _scores([[STEPS[0][0]], [STEPS[1][0]]]),
        pad_token_id=PAD, eos_token_id=EOS,
    )
    batched = inference.compute_transition_scores(
        SEQUENCES, _scores(STEPS), pad_token_id=PAD, eos_token_id=EOS
    )
    assert alone[0] == pytest.approx(batched[0], abs=1e-6)


def test_sequence_without_eos_uses_all_positions_and_several_eos_ids_work() -> None:
    sequences = torch.tensor([[PAD, 2, 3, 2]])  # hit max_length, never emitted EOS
    steps = [[[-10.0, 0.0, 2.0, 0.0]], [[-10.0, 0.0, 0.0, 2.0]], [[-10.0, 0.0, 2.0, 0.0]]]
    expected = sum(_logprob(steps[i], 0, token) for i, token in enumerate([2, 3, 2])) / 3
    for eos in (EOS, [EOS, 5]):
        value = inference.compute_transition_scores(
            sequences, _scores(steps), pad_token_id=PAD, eos_token_id=eos
        )
        assert value[0] == pytest.approx(expected, abs=1e-6)


def test_sum_mode_also_stops_at_eos() -> None:
    values = inference.compute_transition_scores(
        SEQUENCES, _scores(STEPS), pad_token_id=PAD, normalize_by_length=False, eos_token_id=EOS
    )
    assert values[0] == pytest.approx(_logprob(STEPS[0], 0, 2) + _logprob(STEPS[1], 0, EOS), abs=1e-6)


def test_generation_scores_match_textbook_perplexity_for_sequences_of_different_length() -> None:
    """With top-k off, the generation-time score must equal the teacher-forced NLL."""
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    eos_ids = [1, 2, 3, 4, 5]  # many stop tokens -> sequences of very different lengths
    config = transformers.T5Config(
        vocab_size=60, d_model=32, d_ff=64, d_kv=8, num_heads=2, num_layers=1, num_decoder_layers=1,
        decoder_start_token_id=PAD, pad_token_id=PAD, eos_token_id=eos_ids[0],
    )
    model = transformers.T5ForConditionalGeneration(config).eval()
    src = torch.randint(6, 60, (1, 7))
    with torch.no_grad():
        out = model.generate(
            src, do_sample=True, top_k=0, top_p=1.0, max_length=40, num_return_sequences=10,
            eos_token_id=eos_ids, pad_token_id=PAD, return_dict_in_generate=True, output_scores=True,
        )
    sequences = out.sequences
    lengths = [int(((s[1:] != PAD)).sum()) for s in sequences]
    assert len(set(lengths)) > 3, f"need sequences of different length, got {lengths}"

    fixed = inference.compute_transition_scores(sequences, out.scores, PAD, True, eos_token_id=eos_ids)
    old = inference.compute_transition_scores(sequences, out.scores, PAD, True)

    for i, seq in enumerate(sequences):
        tokens = seq[1:]
        eos_positions = [j for j, t in enumerate(tokens.tolist()) if t in eos_ids]
        stop = eos_positions[0] + 1 if eos_positions else len(tokens)
        labels = tokens[:stop].unsqueeze(0)
        with torch.no_grad():
            textbook_nll = model(input_ids=src, labels=labels).loss.item()
        assert fixed[i] == pytest.approx(-textbook_nll, abs=1e-4), (i, lengths[i])
        assert math.exp(-fixed[i]) == pytest.approx(math.exp(textbook_nll), rel=1e-3)

    # the old behaviour deviates for the sequences that finished early
    assert any(abs(f - o) > 1e-3 for f, o in zip(fixed, old))
