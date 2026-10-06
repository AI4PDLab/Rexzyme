# Encoder embedding distance to the training set

Extract Rexzyme encoder embeddings for training reactions and for the held-out
set (0–1255), then compute the minimum cosine distance from each held-out
reaction to the training set. The paper checkpoint, SMILES tokenizer, and
training JSON (`translation.Reaction` / `translation.Sequence` JSONL) default to
[AI4PD/REXzyme](https://huggingface.co/AI4PD/REXzyme).

Defaults match the paper analysis: `--layer-index 0` (encoder embeddings),
attention-weighted pooling on the training set, mean/max/std pooling on the
held-out set.

```bash
python eval/embedding_distance/extract_encoder_embeddings.py \
  --model-path AI4PD/REXzyme/checkpoint-90000 \
  --tokenizer-mol AI4PD/REXzyme/tokenizer_smiles \
  --train-file AI4PD/REXzyme \
  --heldout-file data/generated/generation_heldout/smiles_map.json \
  --output embeddings \
  --layer-index 0 \
  --batch-size 16

python eval/embedding_distance/embedding_distance_to_train.py \
  --train-embeddings embeddings/train_embeddings.pkl \
  --heldout-embeddings embeddings/heldout_embeddings.pkl \
  --smiles-map data/generated/generation_heldout/smiles_map.json \
  --output embeddings/heldout_embedding_distance_to_train.json \
  --output-tsv embeddings/embedding_distance_by_file.tsv
```

The second script also writes `embedding_distance_by_file.tsv` using
[`data/generated/generation_heldout/smiles_map.json`](../../data/generated/generation_heldout/smiles_map.json)
so every `output_file_N.fasta` (including 1255) has a distance.

Optional: pass `--fingerprints-json fingerprint_distances.json` to copy the
fingerprint distance into the JSON.

Attention-weighted training embeddings call `generate` and are slow on a full
training set. Use `--train-pooling pooled` for a faster encoder-only run, or
retry a failed GPU batch with a smaller `--batch-size`.

Paper tables:

- [`../tmscore_plots/tmscore_vs_embed_distance.tsv`](../tmscore_plots/tmscore_vs_embed_distance.tsv) — TM-score vs embedding distance
- [`../tmscore_plots/embed_distance_vs_fp_distance.tsv`](../tmscore_plots/embed_distance_vs_fp_distance.tsv) — embedding vs fingerprint distance
