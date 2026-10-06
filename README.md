# Rexzyme

Code for the paper: **Rexzyme: A Translation Machine for the design of enzymes tailored to specific reactions** (_link to add_).

Rexzyme is an encoder–decoder protein language model that takes a reaction SMILES and generates enzyme sequences.

## Repository layout

| Path | Contents |
|------|----------|
| [`src/`](src/) | Tokenize, train, and generate sequences |
| [`scripts/`](scripts/) | Wrappers for paper inference and example training |
| [`design_specific_enzymes/`](design_specific_enzymes/) | Practical workflow for task-specific enzyme design (fine-tuning + proxy-based DPO) |
| [`data/`](data/) | Paper reaction lists, generated FASTAs, Rhea/EC tables |
| [`eval/`](eval/) | Analyses used in the paper |
| [`tests/`](tests/) | Unit tests for public script contracts |

Detailed notes: [`src/README.md`](src/README.md), [`scripts/README.md`](scripts/README.md), [`design_specific_enzymes/README.md`](design_specific_enzymes/README.md), [`data/README.md`](data/README.md), [`eval/README.md`](eval/README.md), [`tests/README.md`](tests/README.md).

## Requirements

- Python 3.10+
- Core packages in [`requirements.txt`](requirements.txt): `torch`, `transformers`, `datasets`, `numpy`, `huggingface_hub`
- Evaluation extras in [`requirements-eval.txt`](requirements-eval.txt): `pandas`, `matplotlib`, `biopython`, `scipy`, `rdkit`, `tqdm`
- External tools (not installed by pip): **MMseqs**, **Foldseek**, **ESMFold**, and **CLEAN**

```bash
pip install -r requirements.txt
pip install -r requirements-eval.txt
pytest tests
```

## Hugging Face assets

The paper **model**, **tokenizers**, and **train/validation datasets** are on
[Hugging Face: AI4PD/REXzyme](https://huggingface.co/AI4PD/REXzyme):

| Asset | Path in the repo |
|-------|------------------|
| Paper checkpoint | `checkpoint-90000` |
| Amino-acid tokenizer | `tokenizer_aa` |
| SMILES tokenizer | `tokenizer_smiles` |
| Training set | `training_dataset.json` (JSONL) |
| Validation set | `validation_dataset.json` (JSONL) |

Each JSONL line is `{"translation": {"Reaction": "<SMILES>", "Sequence": "<protein>"}}`.

Scripts accept those Hub ids directly. To download the JSON files into the
Hugging Face cache (they are large):

```bash
python scripts/download_hf_assets.py
```

## Generate sequences

**Input:** one reaction SMILES per line (or a JSON/JSONL file with `translation` entries).

**Sampling parameters** (`top_k=100`, `top_p=1.0`, `repetition_penalty=1.0`, 25 sequences per reaction):

```bash
export INPUT_FILE=data/generation_most.txt # change this to your input file containing the reaction(s) you would like to use as input for the generation
export OUTPUT_FOLDER=generation_output/generation_most
bash scripts/run_inference.sh
```


Use the same command with any other list in [`data/`](data/): `generation_middle.txt`, `generation_low.txt`, `generation_val_com.txt`, `generation_val_dif.txt`, `generation_heldout.txt` or a file that you create containing reaction(s) in SMILES format.

Or call the Python script directly:

```bash
python src/inference.py \
  --model_path AI4PD/REXzyme/checkpoint-90000 \
  --tokenizer_aa AI4PD/REXzyme/tokenizer_aa \
  --tokenizer_mol AI4PD/REXzyme/tokenizer_smiles \
  --input_file data/generation_most.txt \
  --output_folder generation_output/generation_most \
  --top_k 100 \
  --top_p 1.0 \
  --repetition_penalty 1.0 \
  --num_return_sequences 25 \
  --seed 0
```

Outputs in `--output_folder`:

- one FASTA per reaction (`output_topk{K}_file_{N}.fasta`, headers `>idx_ppl_...`)
- `molecule_input_metadata.json` — FASTA filename → SMILES mapping
- `inference_metadata.json` — per-reaction perplexities and optional ground-truth
- `generation_parameters.json` — full record of the generation settings used

The sequences used in the paper are already in [`data/generated/`](data/generated/).

## Reaction groups (in the paper)

| Group | Reactions | Training support |
|-------|-----------|------------------|
| [`data/generation_most.txt`](data/generation_most.txt) | 20 | >10,000 sequences / reaction |
| [`data/generation_middle.txt`](data/generation_middle.txt) | 20 | 100–10,000 |
| [`data/generation_low.txt`](data/generation_low.txt) | 20 | <100 |
| [`data/generation_val_com.txt`](data/generation_val_com.txt) | 20 | 0, same EC in training |
| [`data/generation_val_dif.txt`](data/generation_val_dif.txt) | 20 | 0, same EC not in training |
| [`data/generation_heldout.txt`](data/generation_heldout.txt) | 1,256 (indices 0–1255) | held-out Rhea reactions |

Rhea IDs and EC labels: [`data/extended_reaction_information/`](data/extended_reaction_information/).

## Evaluation

Scripts under [`eval/`](eval/) reproduce the paper figures. Group-level plots (pLDDT, CLEAN, training-set identity) use the five small `generation_*` sets. Structural generalisation uses the held-out set (reactions 0–1255).

### Group-level analyses

1. Fold with ESMFold and plot pLDDT: [`eval/plddt/`](eval/plddt/)
2. CLEAN EC prediction and measure of accuracy at levels 1–4: [`eval/clean/`](eval/clean/)
3. Sequence identity to the training set (MMseqs): [`eval/train_identity/`](eval/train_identity/)
4. Recovery of known ground-truth sequences: [`eval/gt_acc/`](eval/gt_acc/)

### Held-out generalisation

1. Download natural structures and superpose designs (folded with ESMFold) with Foldseek: [`eval/gt_superpose/`](eval/gt_superpose/)
2. Fingerprint distance held out to training reactions: [`eval/fp_distance/`](eval/fp_distance/)
3. Sequence identity of designs to the training set: [`eval/designs_vs_train/`](eval/designs_vs_train/)
4. Encoder embedding distance between held out and training reactions: [`eval/embedding_distance/`](eval/embedding_distance/)
5. Plot TM-score vs pLDDT, vs fingerprint distance, vs sequence identity, and vs embedding distance: [`eval/tmscore_plots/`](eval/tmscore_plots/)

Tables for step 5 (reactions 0–1255) used in the paper are found here: `tmscore_vs_plddt.tsv`, `tmscore_vs_fp_distance.tsv`, `tmscore_vs_seq_id.tsv`, `tmscore_vs_embed_distance.tsv`, and `embed_distance_vs_fp_distance.tsv` in that folder.

Training and validation JSON are downloaded from [AI4PD/REXzyme](https://huggingface.co/AI4PD/REXzyme) when a script needs them (`--train-json AI4PD/REXzyme` is the default). A training FASTA, if you need one for MMseqs, is built from that JSON with [`eval/train_identity/mmseq_file_preparation.py`](eval/train_identity/mmseq_file_preparation.py). Commands and layouts: [`eval/README.md`](eval/README.md).

## Training

Tokenize the Hub train/validation JSON and train:

```bash
export TOKENIZER_AA=AI4PD/REXzyme/tokenizer_aa
export TOKENIZER_MOL=AI4PD/REXzyme/tokenizer_smiles
export MODEL_NAME_OR_PATH=AI4PD/REXzyme/checkpoint-90000
export DATASET=AI4PD/REXzyme
bash scripts/run_train.sh
```

See [`src/README.md`](src/README.md).

## Citation

If you find this work useful, please cite:

_To Do_
