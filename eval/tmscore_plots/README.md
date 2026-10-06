# Held-out TM-score plots

After [Foldseek superposition](../gt_superpose/README.md), compare each design's
best TM-score to ground truth with pLDDT, fingerprint distance, sequence
identity, and encoder embedding distance. Defaults cover the full held-out set
(reactions **0–1255**). See the parent [`../README.md`](../README.md) for how
this step sits in the overall evaluation.

The three distance plots keep designs with **mean pLDDT ≥ 70**, then one point
per reaction (the remaining design with the highest TM-score). Build the
all-designs pLDDT table first; do **not** pass `--best-per-reaction` for that
step.

## 1. TM-score versus pLDDT

```bash
python eval/tmscore_plots/plot_tmscore_vs_plddt.py \
  --master /path/to/superposition_ground_truth_designs/all_tmscores.tsv \
  --designs /path/to/designs \
  --out-prefix tmscore_vs_plddt
```

`--designs` is the same per-reaction PDB layout as superposition
(`output_file_<N>/pdbs/*.pdb`). Plot scripts match both `output_file_N` and
`output_file-N` names.

## 2. TM-score versus fingerprint distance

Compute distances first ([`eval/fp_distance/`](../fp_distance/README.md)):

```bash
bash eval/fp_distance/run_fp_distance.sh \
  --smiles-map data/generated/generation_heldout/smiles_map.json \
  --output fingerprint_distances.json

python eval/tmscore_plots/plot_tmscore_vs_fp_distance.py \
  --plddt-tsv tmscore_vs_plddt.tsv \
  --fp fingerprint_distances.json \
  --out-prefix tmscore_vs_fp_distance
```

## 3. TM-score versus sequence identity

Search designs against the training sequences
([`eval/designs_vs_train/`](../designs_vs_train/README.md)):

```bash
python eval/train_identity/mmseq_file_preparation.py \
  --json_file AI4PD/REXzyme \
  --output training_dataset.fasta
bash eval/designs_vs_train/run_mmseqs_designs_vs_train.sh \
  --query data/generated/generation_heldout/all_sequences_with_filenames.fasta \
  --target training_dataset.fasta \
  --output designs_vs_train

python eval/tmscore_plots/plot_tmscore_vs_seq_id.py \
  --plddt-tsv tmscore_vs_plddt.tsv \
  --mmseqs designs_vs_train/designs_vs_train.best.tsv \
  --out-prefix tmscore_vs_seq_id
```

## 4. TM-score versus encoder embedding distance

Extract embeddings and nearest-neighbour distances
([`eval/embedding_distance/`](../embedding_distance/README.md)):

```bash
python eval/embedding_distance/extract_encoder_embeddings.py \
  --model-path AI4PD/REXzyme/checkpoint-90000 \
  --tokenizer-mol AI4PD/REXzyme/tokenizer_smiles \
  --train-file AI4PD/REXzyme \
  --output embeddings

python eval/embedding_distance/embedding_distance_to_train.py \
  --train-embeddings embeddings/train_embeddings.pkl \
  --heldout-embeddings embeddings/heldout_embeddings.pkl \
  --output embeddings/heldout_embedding_distance_to_train.json

python eval/tmscore_plots/plot_tmscore_vs_embed_distance.py \
  --plddt-tsv tmscore_vs_plddt.tsv \
  --embedding-tsv embeddings/embedding_distance_by_file.tsv \
  --out-prefix tmscore_vs_embed_distance
```

Each plot writes `<out-prefix>.tsv`, `.png`, and `.pdf`. The paper tables
listed below are already in this folder; rerun the scripts only if you are
reproducing the figures from Foldseek and distance files.

## Paper tables

Held-out results (reactions **0–1255**) matching those `--out-prefix` names:

| File | Script | Rows |
|------|--------|------|
| [`tmscore_vs_plddt.tsv`](tmscore_vs_plddt.tsv) | [`plot_tmscore_vs_plddt.py`](plot_tmscore_vs_plddt.py) | all designs |
| [`tmscore_vs_fp_distance.tsv`](tmscore_vs_fp_distance.tsv) | [`plot_tmscore_vs_fp_distance.py`](plot_tmscore_vs_fp_distance.py) | one per reaction, pLDDT ≥ 70 |
| [`tmscore_vs_seq_id.tsv`](tmscore_vs_seq_id.tsv) | [`plot_tmscore_vs_seq_id.py`](plot_tmscore_vs_seq_id.py) | one per reaction, pLDDT ≥ 70 |
| [`tmscore_vs_embed_distance.tsv`](tmscore_vs_embed_distance.tsv) | [`plot_tmscore_vs_embed_distance.py`](plot_tmscore_vs_embed_distance.py) | one per reaction, pLDDT ≥ 70 |

[`embed_distance_vs_fp_distance.tsv`](embed_distance_vs_fp_distance.tsv) is the same pLDDT ≥ 70 reaction set with fingerprint distance (`min_distance`) and encoder embedding distance (`min_emb_distance`).
