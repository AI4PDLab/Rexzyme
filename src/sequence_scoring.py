"""Teacher-forced log-likelihood of sequences returned by ``model.generate()``.

``generate()`` returns every row as a *prefix* the model was conditioned on, followed by
the generated tokens::

    encoder-decoder (T5, BART, ...):  [decoder_start, t1, ..., tn, eos, pad, ...]
    decoder-only (GPT-2, ...):        [prompt ...,    t1, ..., tn, eos, pad, ...]

Passing that output back as ``labels`` scores it wrongly. Encoder-decoder models shift
``labels`` right and prepend ``decoder_start_token_id`` again, so the decoder is conditioned
on two start tokens (``[start, start, t1, ...]``); if the start token is not the pad id it is
also scored as a target. For decoder-only models ``labels`` does not even match the length of
``input_ids``.

``sequence_log_likelihood`` instead gives the model exactly the context it had while
generating (``sequences[:, :-1]``) and scores ``sequences[:, prefix_length:]``. It:

* removes the prefix by position, not by token id, so it does not matter whether the start
  token equals the pad or the EOS id;
* stops at the first EOS of each row, so models whose pad id equals the EOS id work;
* computes log-probabilities from the logits, so custom models whose ``loss`` adds extra
  terms (e.g. a contact head) are scored on the language-model term only.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

ENCODER_DECODER = "encoder_decoder"
DECODER_ONLY = "decoder_only"


def _config_value(model, name: str):
    for config in (getattr(model, "generation_config", None), getattr(model, "config", None)):
        value = getattr(config, name, None) if config is not None else None
        if value is not None:
            return value
    return None


def _id_tensor(ids, device) -> torch.Tensor | None:
    if ids is None:
        return None
    tensor = torch.as_tensor(ids, device=device).reshape(-1)
    return tensor if tensor.numel() else None


def _expand_batch(value: Any, batch_size: int) -> Any:
    """Repeat a batch-size-1 tensor ``batch_size`` times (other values are returned as is)."""
    if batch_size > 1 and torch.is_tensor(value) and value.dim() > 0 and value.size(0) == 1:
        return value.expand(batch_size, *value.shape[1:])
    return value


def generated_token_mask(targets: torch.Tensor, eos_token_id=None, pad_token_id=None) -> torch.Tensor:
    """Positions of ``targets`` (generated tokens only) that are scored.

    Every position up to and including the first EOS of each row is kept; what follows is the
    padding ``generate()`` appends once a row has finished. Only when no EOS id is known are
    pad tokens used instead to find that padding. Tokens before the EOS are always scored, even
    if a model happens to generate the pad id, because they are part of the sequence.
    """
    mask = torch.ones_like(targets, dtype=torch.bool)
    eos_ids = _id_tensor(eos_token_id, targets.device)
    if eos_ids is not None:
        is_eos = torch.isin(targets, eos_ids).long()
        mask &= (is_eos.cumsum(dim=1) - is_eos) == 0  # no EOS strictly before this position
    elif pad_token_id is not None:
        mask &= targets != pad_token_id
    return mask


def sequence_log_likelihood(
    model,
    sequences: torch.Tensor,
    *,
    prefix_length: int | None = None,
    mode: str | None = None,
    model_inputs: Mapping[str, Any] | None = None,
    attention_mask: torch.Tensor | None = None,
    eos_token_id=None,
    pad_token_id=None,
):
    """Log-likelihood of the generated part of each row of ``sequences``.

    Args:
        model: the model that produced ``sequences``.
        sequences: ``(B, T)`` output of ``generate()`` (its ``.sequences`` when it returns a dict).
        prefix_length: leading tokens of each row that were not generated. Defaults to 1 for
            encoder-decoder models (the decoder start token); required for decoder-only models
            (the prompt length).
        mode: ``"encoder_decoder"`` or ``"decoder_only"``. Defaults to
            ``model.config.is_encoder_decoder``; pass it explicitly for custom models that do
            not set that flag.
        model_inputs: other keyword arguments for ``model(...)``: the encoder inputs
            (``input_ids``, ``attention_mask``, ...) of an encoder-decoder model, or the
            conditioning inputs of a decoder-only model. Tensors with batch size 1 are repeated
            to batch size ``B``.
        attention_mask: decoder-only models: ``(B or 1, prefix_length)`` mask of the prompt.
        eos_token_id, pad_token_id: default to the model's generation config, then its config.

    Returns:
        ``(log_likelihood, n_tokens, outputs)``: the summed log-probability ``(B,)`` and the
        number of scored tokens ``(B,)`` of each row, and the raw model outputs. Position ``j``
        of ``outputs.logits`` (and of any other per-position output) predicts
        ``sequences[:, j + 1]``.
    """
    if mode is None:
        mode = ENCODER_DECODER if getattr(model.config, "is_encoder_decoder", False) else DECODER_ONLY
    if mode not in (ENCODER_DECODER, DECODER_ONLY):
        raise ValueError(f"mode must be {ENCODER_DECODER!r} or {DECODER_ONLY!r}, got {mode!r}")
    if prefix_length is None:
        if mode == DECODER_ONLY:
            raise ValueError("prefix_length (the prompt length) is required for decoder-only models")
        prefix_length = 1
    if prefix_length < 1:
        raise ValueError("prefix_length must be at least 1: generation always starts from a prefix")
    if eos_token_id is None:
        eos_token_id = _config_value(model, "eos_token_id")
    if pad_token_id is None:
        pad_token_id = _config_value(model, "pad_token_id")

    batch_size, length = sequences.shape
    if length <= prefix_length:  # nothing was generated
        zeros = torch.zeros(batch_size, device=sequences.device)
        return zeros, zeros.long(), None

    kwargs = {name: _expand_batch(value, batch_size) for name, value in (model_inputs or {}).items()}
    context = sequences[:, :-1]  # what the model had seen when it produced each next token
    if mode == ENCODER_DECODER:
        kwargs["decoder_input_ids"] = context
    else:
        kwargs["input_ids"] = context
        if attention_mask is not None:
            prompt_mask = _expand_batch(attention_mask, batch_size).to(context.device)
            generated = torch.ones(
                batch_size, context.size(1) - prompt_mask.size(1), dtype=prompt_mask.dtype, device=context.device
            )
            kwargs["attention_mask"] = torch.cat([prompt_mask, generated], dim=1)

    outputs = model(**kwargs)
    logits = outputs.logits[:, prefix_length - 1 :]
    targets = sequences[:, prefix_length:]
    mask = generated_token_mask(targets, eos_token_id, pad_token_id)
    log_probs = torch.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    log_probs = log_probs.masked_fill(~mask, 0.0)
    return log_probs.sum(dim=1), mask.sum(dim=1), outputs


def _rows(value: Any, start: int, end: int, batch_size: int) -> Any:
    """Rows ``start:end`` of a per-row tensor (batch dimension ``batch_size``); others as is."""
    if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == batch_size and batch_size > 1:
        return value[start:end]
    return value


@torch.no_grad()
def sequence_perplexities(
    model, sequences: torch.Tensor, *, batch_size: int | None = None, **kwargs
) -> list[float]:
    """Per-row perplexity ``exp(-mean log-prob)`` of the generated tokens (NaN if none).

    Takes the keyword arguments of :func:`sequence_log_likelihood`. ``batch_size`` scores the
    rows in chunks (default: all at once). Teacher forcing ``B`` rows of length ``T`` needs
    attention memory proportional to ``B * T**2``, so long sequences may need small chunks.
    """
    total_rows = sequences.size(0)
    step = batch_size or total_rows
    perplexities: list[float] = []
    for start in range(0, total_rows, step):
        end = min(start + step, total_rows)
        chunk_kwargs = dict(kwargs)
        if chunk_kwargs.get("model_inputs") is not None:
            chunk_kwargs["model_inputs"] = {
                name: _rows(value, start, end, total_rows) for name, value in chunk_kwargs["model_inputs"].items()
            }
        if chunk_kwargs.get("attention_mask") is not None:
            chunk_kwargs["attention_mask"] = _rows(chunk_kwargs["attention_mask"], start, end, total_rows)
        log_likelihood, n_tokens, _ = sequence_log_likelihood(model, sequences[start:end], **chunk_kwargs)
        perplexities += [
            math.exp(-total / count) if count else float("nan")
            for total, count in zip(log_likelihood.tolist(), n_tokens.tolist())
        ]
    return perplexities
