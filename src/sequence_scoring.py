"""Perplexity of sequences returned by T5 ``model.generate()``.

``generate()`` returns every row as ``[decoder_start, t1, ..., tn, eos, pad, ...]``. Passed back
as ``labels``, such a row is shifted right by T5, which prepends the decoder start token again, so
the decoder would be conditioned on ``[start, start, t1, ...]``. ``generated_sequence_nll`` drops
the start token first, so every token is scored in the context it was generated in (and the same
context as in training).
"""

import torch


@torch.no_grad()
def generated_sequence_nll(model, sequences, input_ids, attention_mask=None):
    """Mean negative log-likelihood per generated token, for each row of ``sequences``.

    ``input_ids`` / ``attention_mask`` are the encoder inputs (batch size 1) that ``generate()``
    was called with. Positions after the first EOS of a row (padding) are not scored. Rows are
    scored one at a time, which keeps peak GPU memory as low as during generation.
    """
    nlls = []
    for row in sequences:
        labels = row[1:].unsqueeze(0).clone()  # drop the decoder start token
        is_eos = (labels == model.config.eos_token_id).long()
        labels[(is_eos.cumsum(dim=1) - is_eos) > 0] = -100  # positions after the first EOS
        nlls.append(model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss.item())
    return nlls
