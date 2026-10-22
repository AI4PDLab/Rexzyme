# Designing Specific Enzymes with Rexzyme

This folder documents a practical workflow to adapt Rexzyme to a specific design
goal:

1. Fine-tuning (all weights)
2. Reinforcement learning
3. In silico metrics

This version covers:

- **Part I: fine-tuning**
- **Part II (DPO-style preference optimization by custom proxy)**
- **Part III (multi-stage in silico filtering pipeline)**

## Part I - Fine-tuning (all weights)

Fine-tuning means standard supervised training on our own reaction-enzyme pairs,
starting from pretrained Rexzyme weights.

## 1) Prepare our dataset (JSONL)

Use JSONL lines with the schema:

```json
{"translation": {"Reaction": "C(=O)..", "Sequence": "MS..."}}
{"translation": {"Reaction": "C(=O)..", "Sequence": "MS..."}}
```
The sequences we used for the fine-tuning of FDHs descrived in the manuscript can be found under `finetunning_sequences/`

Recommended: provide **two files in the same directory**, named exactly:

- `train.json` (or `train.jsonl`)
- `val.json` (or `val.jsonl`)

`src/tokenize.py` auto-detects those names when `--dataset` points to a
directory. Other names (e.g. `training_dataset.json`) are **not** auto-detected
— use `--data_files` to point to them explicitly (see Option A below).

If only one file is provided, the entire file becomes the `train` split and
training proceeds without a validation set (a warning is printed).

## 2) Tokenize our custom data

Use the tokenization scripts provided in this repository.

### Option A - Use `src/tokenize.py` directly

If our data is already split into `train.json` / `val.json` files in one directory:

```bash
python -m src.tokenize \
  --dataset /path/to/your_splits_dir \
  --tokenizer_aa AI4PD/REXzyme/tokenizer_aa \
  --tokenizer_mol AI4PD/REXzyme/tokenizer_smiles \
  --output_dir /path/to/tokenized_dataset
```

If the files have other names, pass them explicitly with `--data_files`:

```bash
python -m src.tokenize \
  --dataset /path/to/your_splits_dir \
  --data_files train=training_dataset.json \
  --data_files validation=validation_dataset.json \
  --tokenizer_aa AI4PD/REXzyme/tokenizer_aa \
  --tokenizer_mol AI4PD/REXzyme/tokenizer_smiles \
  --output_dir /path/to/tokenized_dataset
```

Notes:

- `--dataset` accepts a local JSON/JSONL file, a split directory, or a Hub id.
- Tokenizer paths support Hub subfolder notation (`AI4PD/REXzyme/tokenizer_aa`)
  or a local directory.
- Tokenization output is written to `--output_dir` and consumed by `src/train.py`.

### Option B - Use the wrapper script

```bash
export TOKENIZER_AA=/path/to/tokenizer_aa
export TOKENIZER_MOL=/path/to/tokenizer_smiles
export MODEL_NAME_OR_PATH=/path/to/pretrained_rexzyme_checkpoint
export LOCAL_DATASET_PATH=/path/to/your_splits_dir
export TOKENIZED_DIR=/path/to/tokenized_dataset
export OUTPUT_DIR=/path/to/training_output
bash scripts/run_train.sh
```

## 3) Train (fine-tune all weights)

Critical points for fine-tuning:

- Set `--model_name_or_path` (or `MODEL_NAME_OR_PATH`) to the pretrained
  checkpoint path (e.g. `AI4PD/REXzyme/checkpoint-90000`).

Fine-tuning from a pretrained checkpoint is the default behavior of
`src/train.py`.

Direct example:

```bash
python src/train.py \
  --tokenized_dir /path/to/tokenized_dataset \
  --model_name_or_path /path/to/pretrained_rexzyme_checkpoint \
  --output_dir /path/to/training_output
```

## Next parts

- Part IV (broader reinforcement learning): to be added

## Part II - DPO by custom proxy (weighted)

Use [`dpo_proxy/`](dpo_proxy/) to optimize a finetuned encoder-decoder model
with preference-style updates and then generate sequences.

Main script:

- [`dpo_proxy/dpo_optimize_and_generate.py`](dpo_proxy/dpo_optimize_and_generate.py)

It supports:

- the weighted objective (`--mode weighted`, the default and only mode)
- any user-defined numeric proxy column from a CSV/TSV/XLSX file
- objective direction control (`--objective maximize|minimize`)

Typical proxy examples:

- catalytic values (`Kcat`, `Km`, `Kcat/Km`)
- structure/quality proxies (`plddt`, `tmscore`)
- any custom score we provide in the table

## Part III - Multi-stage filtering pipeline

Use [`filtering_pipeline/`](filtering_pipeline/) for a practical sequence
selection workflow after generation.

It includes the full stage order we use:

1. deduplicate
2. perplexity top 25%
3. CLEAN EC match to target
4. AlphaFold + pLDDT > 75 (oligomer-aware)
5. MMseqs remove 100% natural identity
6. ipTM top 25%
7. Foldseek top TM-score (for example top 1000)
8. ESM1v top subset (for example 500)
9. ProteinMPNN top subset (for example 200)
10. PLIP + MSA interaction/catalytic checks
11. Rosetta metrics bounds
