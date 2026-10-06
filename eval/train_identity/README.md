# Sequence identity to the training set (group plots)

MMseqs identity of generated sequences versus the Rexzyme training set, for
the five small `generation_*` groups. Held-out designs use
[`../designs_vs_train/`](../designs_vs_train/) instead.

The training JSON is on [Hugging Face: AI4PD/REXzyme](https://huggingface.co/AI4PD/REXzyme)
(`training_dataset.json`: JSONL with `translation.Reaction` and
`translation.Sequence`). [`mmseq_file_preparation.py`](mmseq_file_preparation.py)
reads that format (or a JSON array; `Protein sequence` is a fallback).

## 1. Prepare FASTAs

```bash
python eval/train_identity/mmseq_file_preparation.py \
  --json_file AI4PD/REXzyme \
  --output training_dataset.fasta
```

`--fasta_folder` collates generated FASTAs into a numeric-header query file
(optional; used with the interpretation script).

## 2. Search and summarise

Run MMseqs against `training_dataset.fasta`, then:

```bash
python eval/train_identity/mmseq_interpretation.py \
  --mmseqs_path /path/to/mmseqs_dir \
  --query_fasta /path/to/query.fasta \
  --output-dir /path/to/identity_results
```

## 3. Plot identity versus alignment length

```bash
python eval/train_identity/plot_mmseqs_identity_vs_alignment.py \
  --input /path/to/mmseqs_output \
  --output mmseqs_hexbin \
  --title ">10,000 sequences per reaction" \
  --color "#68AAA3"
```

The input must contain at least: query, target, sequence identity, alignment
length.
