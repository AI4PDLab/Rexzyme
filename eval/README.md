# Evaluation

Scripts to reproduce the paper analyses after generating sequences with
[`src/inference.py`](../src/inference.py) (or using the FASTAs already in
[`data/generated/`](../data/generated/)).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-eval.txt
```

MMseqs and Foldseek are system dependencies (not pip packages). Put
`mmseqs` / `foldseek` on `PATH`, or pass an explicit executable path.
CLEAN and ESMFold are also external.

Every script takes input and output paths on the command line
(`--input` / `--output`, or more specific names such as `--fasta_path`,
`--query`, `--designs`). Run any script with `-h` to see the flags.

## Script index

| Folder | What it does |
|--------|----------------|
| [`plddt/`](plddt/) | Fold designs (ESMFold) and plot pLDDT by group |
| [`clean/`](clean/) | Run CLEAN on FASTAs and plot EC accuracy at levels 1–4 |
| [`train_identity/`](train_identity/) | MMseqs identity of generated sequences vs the training set |
| [`gt_acc/`](gt_acc/) | MMseqs recovery of known ground-truth enzymes at sequence level |
| [`gt_superpose/`](gt_superpose/) | Download natural structures and Foldseek-superpose held-out designs |
| [`fp_distance/`](fp_distance/) | Fingerprint distance from held-out reactions to training reactions |
| [`designs_vs_train/`](designs_vs_train/) | MMseqs of all held-out designs vs training sequences |
| [`embedding_distance/`](embedding_distance/) | Encoder embeddings and nearest-neighbour distance to training reactions |
| [`tmscore_plots/`](tmscore_plots/) | TM-score plots and TSV tables |

## Reaction groups

Each file in [`data/`](../data/) is one SMILES per line. Inference generates
25 sequences per reaction.

| Group | Reaction file | Training dataset |
|-------|---------------|------------------|
| `generation_most` | [`data/generation_most.txt`](../data/generation_most.txt) | >10,000 sequences / reaction |
| `generation_middle` | [`data/generation_middle.txt`](../data/generation_middle.txt) | 100–10,000 |
| `generation_low` | [`data/generation_low.txt`](../data/generation_low.txt) | <100 |
| `generation_val_com` | [`data/generation_val_com.txt`](../data/generation_val_com.txt) | 0, same EC in training |
| `generation_val_dif` | [`data/generation_val_dif.txt`](../data/generation_val_dif.txt) | 0, same EC not in training |
| `generation_heldout` | [`data/generation_heldout.txt`](../data/generation_heldout.txt) | held-out Rhea reactions (0–1255) |

Paper sequences: [`data/generated/`](../data/generated/). Rhea / EC tables:
[`data/extended_reaction_information/`](../data/extended_reaction_information/).

Training and validation JSON are on
[Hugging Face: AI4PD/REXzyme](https://huggingface.co/AI4PD/REXzyme).
They are JSONL files (`{"translation": {"Reaction": "...", "Sequence": "..."}}`).
Scripts that need them default to that repo (`--train-json AI4PD/REXzyme`)
and download `training_dataset.json` via `huggingface_hub`. Or run
[`scripts/download_hf_assets.py`](../scripts/download_hf_assets.py) first.

## Held-out pipeline (reactions 0–1255)

End-to-end structural generalisation, after sequences exist in
[`data/generated/generation_heldout/`](../data/generated/generation_heldout/):

1. Fold designs ([`plddt/run-esm.py`](plddt/run-esm.py)) into
   `designs/output_file_<N>/pdbs/*.pdb`.
2. Download ground-truth structures and superpose with Foldseek:
   [`gt_superpose/README.md`](gt_superpose/README.md).
3. Compute fingerprint distance, design-vs-train identity, and encoder
   embedding distance (each folder has its own README).
4. Plot TM-score against those metrics:
   [`tmscore_plots/README.md`](tmscore_plots/README.md).
   Paper tables are already in [`tmscore_plots/`](tmscore_plots/).

The first TM-score plot (vs pLDDT) must include **all** designs. The other
three keep designs with mean pLDDT ≥ 70, then one point per reaction.

## 1. Structure confidence (pLDDT)

See [`plddt/README.md`](plddt/README.md).

1. Fold generated FASTAs with [`plddt/run-esm.py`](plddt/run-esm.py).
2. Plot mean ESMFold pLDDT by group with [`plddt/plot_plddt_by_group.py`](plddt/plot_plddt_by_group.py).

```bash
python eval/plddt/plot_plddt_by_group.py \
  --fold-dir fold \
  --output plddt_by_training_group
```

[`plddt/plot_plddt.py`](plddt/plot_plddt.py) is a per-folder scatter of average
pLDDT from a single ESMFold output directory.

## 2. CLEAN EC accuracy

See [`clean/README.md`](clean/README.md).

Run CLEAN on generated FASTAs with [`clean/CLEAN_infer_fasta.py`](clean/CLEAN_infer_fasta.py)
(from inside `CLEAN/app`), then plot accuracy at EC levels 1, 1–2, 1–3, and
1–4:

```bash
cd /path/to/CLEAN/app
python /path/to/Rexzyme_nuria/eval/clean/CLEAN_infer_fasta.py \
  --input-fasta /path/to/Rexzyme_nuria/data/generated/generation_most/output_topk100_file_1.fasta
```

Then generate the group-level plot:

```bash
python eval/clean/plot_clean_ec_accuracy.py \
  --most-csv most_predictions.csv \
  --most-ec data/extended_reaction_information/generation_most/generation_most.tsv \
  --output clean_ec_accuracy
```

Pass `--<group>-csv` and `--<group>-ec` for each group you want to include
(`most`, `middle`, `low`, `val-com`, `val-dif`).

## 3. Sequence identity to the training set (MMseqs)

See [`train_identity/README.md`](train_identity/README.md).

1. Prepare query/target FASTAs with [`train_identity/mmseq_file_preparation.py`](train_identity/mmseq_file_preparation.py) (`--json_file AI4PD/REXzyme` downloads the training JSON).
2. Run MMseqs and summarize hits with [`train_identity/mmseq_interpretation.py`](train_identity/mmseq_interpretation.py).
3. Plot identity versus alignment length with [`train_identity/plot_mmseqs_identity_vs_alignment.py`](train_identity/plot_mmseqs_identity_vs_alignment.py).

```bash
python eval/train_identity/plot_mmseqs_identity_vs_alignment.py \
  --input /path/to/mmseqs_output \
  --output mmseqs_hexbin \
  --title ">10,000 sequences per reaction" \
  --color "#68AAA3"
```

The input file must contain at least: query, target, sequence identity,
alignment length.

## 4. Ground-truth sequence recovery

[`gt_acc/`](gt_acc/) compares generated sequences to known ground-truth enzymes
with MMseqs (separate from the training-set identity plots above). See
[`gt_acc/README.md`](gt_acc/README.md).

## 5. Held-out structure superposition

After folding the held-out designs, download natural ground-truth structures
and superpose them with Foldseek (reactions 0–1255):

1. [`gt_superpose/download_gt_pdbs.py`](gt_superpose/download_gt_pdbs.py)
2. [`gt_superpose/superpose_designs_to_gt.sh`](gt_superpose/superpose_designs_to_gt.sh)
3. [`gt_superpose/apply_reliability_filter.py`](gt_superpose/apply_reliability_filter.py)

See [`gt_superpose/README.md`](gt_superpose/README.md).

## 6. Held-out TM-score analyses

After superposition, plot TM-score against pLDDT, fingerprint distance,
sequence identity, and encoder embedding distance:

- [`tmscore_plots/`](tmscore_plots/)
- [`fp_distance/`](fp_distance/)
- [`designs_vs_train/`](designs_vs_train/)
- [`embedding_distance/`](embedding_distance/)

Paper tables (same names as the plot-script `--out-prefix` outputs) are in
[`tmscore_plots/`](tmscore_plots/): `tmscore_vs_plddt.tsv`,
`tmscore_vs_fp_distance.tsv`, `tmscore_vs_seq_id.tsv`,
`tmscore_vs_embed_distance.tsv`, and `embed_distance_vs_fp_distance.tsv`.
See [`tmscore_plots/README.md`](tmscore_plots/README.md).
