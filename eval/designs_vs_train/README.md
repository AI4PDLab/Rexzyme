# Held-out designs versus the training set

Search all generated held-out sequences against the training sequences with
MMseqs (reactions 0–1255, 25 designs each).

The query FASTA is already in the repository:
[`data/generated/generation_heldout/all_sequences_with_filenames.fasta`](../../data/generated/generation_heldout/all_sequences_with_filenames.fasta).

Build a training FASTA from the Hugging Face training JSON (JSONL with
`translation.Reaction` and `translation.Sequence`):

```bash
python eval/train_identity/mmseq_file_preparation.py \
  --json_file AI4PD/REXzyme \
  --output training_dataset.fasta
```

Then:

```bash
bash eval/designs_vs_train/run_mmseqs_designs_vs_train.sh \
  --query data/generated/generation_heldout/all_sequences_with_filenames.fasta \
  --target training_dataset.fasta \
  --output designs_vs_train
```

Outputs:

- `designs_vs_train.m8` — all hits
- `designs_vs_train.best.tsv` — highest bit-score hit per design (`design`, `pident`, …)

Headers in the query FASTA look like `output_file_N_K_ppl_...`.

The paper TM-score versus sequence-identity table is
[`../tmscore_plots/tmscore_vs_seq_id.tsv`](../tmscore_plots/tmscore_vs_seq_id.tsv).
