# Ground-truth sequence recovery

Compare generated sequences to known enzymes for the same reactions (MMseqs),
separate from identity to the **training set**
([`../train_identity/`](../train_identity/),
[`../designs_vs_train/`](../designs_vs_train/)).

Ground-truth proteins come from a dataset JSON (local file, or the Hub train/val
JSON on [AI4PD/REXzyme](https://huggingface.co/AI4PD/REXzyme) if those records
include the reactions you generated).

Inference metadata (`inference_metadata.json` or `molecule_input_metadata.json`)
maps each generated FASTA back to its reaction SMILES.

## Scripts

| Script | Role |
|--------|------|
| [`gt_inference_mmseq.py`](gt_inference_mmseq.py) | Build GT FASTAs from the dataset JSON and search generated FASTAs with MMseqs |
| [`gt_mmseq_interpretation_balanced.py`](gt_mmseq_interpretation_balanced.py) | Per-target mean / max identity across the 25 generations |
| [`gt_permutation_mmseq_eval.py`](gt_permutation_mmseq_eval.py) | Cross-reaction permutation control (GT of other molecules) |

## 1. Build GT FASTAs and run MMseqs

```bash
python eval/gt_acc/gt_inference_mmseq.py \
  --input data/generation_most.txt \
  --inference_folder data/generated/generation_most \
  --dataset_json AI4PD/REXzyme \
  --output gt_fastas
```

`--inference_folder` is the folder of generated FASTAs (and optional
`molecule_input_metadata.json`). `--output` / `--output_folder_gt` is where
GT FASTAs and MMseqs hits are written.

## 2. Summarise balanced identity

```bash
python eval/gt_acc/gt_mmseq_interpretation_balanced.py \
  --mmseqs_path gt_fastas \
  --inference_folder data/generated/generation_most \
  --gt_fasta_folder gt_fastas \
  --output-dir gt_acc_results
```

## 3. Permutation control

```bash
python eval/gt_acc/gt_permutation_mmseq_eval.py \
  --inference_folder data/generated/generation_most \
  --gt_fasta_folder gt_fastas \
  --output-dir gt_permutation_results
```
