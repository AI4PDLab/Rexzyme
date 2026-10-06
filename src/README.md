# Source

Tokenize a reaction–sequence dataset, train a seq2seq model, and generate
enzyme FASTAs. Evaluation scripts live under [`../eval/`](../eval/) and expect
FASTAs plus metadata already on disk.

## Scripts

| Script | Role |
|--------|------|
| [`hf_assets.py`](hf_assets.py) | Resolve files from [AI4PD/REXzyme](https://huggingface.co/AI4PD/REXzyme) |
| [`tokenize.py`](tokenize.py) | Hub-first or local JSON → tokenized `DatasetDict` |
| [`train.py`](train.py) | `Seq2SeqTrainer` on that `DatasetDict` |
| [`inference.py`](inference.py) | Generate FASTAs from reaction SMILES |

Wrappers: [`../scripts/run_inference.sh`](../scripts/run_inference.sh) (paper sampling) and [`../scripts/run_train.sh`](../scripts/run_train.sh).

The paper checkpoint (`checkpoint-90000`), tokenizers (`tokenizer_aa`,
`tokenizer_smiles`), and train/validation JSON live in the same Hub repo.
Tokenizers are Hugging Face `AutoTokenizer` checkpoints (Hub ids or local directories).

## Generate sequences

[`inference.py`](inference.py) reads a SMILES text file (or JSON/JSONL with
`translation` entries) and writes one FASTA per reaction. Paper generation
used `top_k=100`, `top_p=1.0`, `repetition_penalty=1.0`, and 25 sequences:

```bash
export INPUT_FILE=data/generation_most.txt
export OUTPUT_FOLDER=generation_output
bash scripts/run_inference.sh
```

FASTA headers are `>idx_ppl_...` (teacher-forced conditional perplexity).
The wrapper defaults to `AI4PD/REXzyme/checkpoint-90000`. See
[`../scripts/README.md`](../scripts/README.md) for every environment variable.

Optional flags on `inference.py`: `--top_k`, `--top_p`, `--repetition_penalty`,
`--num_return_sequences`, `--seed`.

## Dataset loading (tokenize)

[`tokenize.py`](tokenize.py) uses Hugging Face `datasets.load_dataset`:

| Source | `--dataset` value | Behavior |
|--------|-------------------|----------|
| **Hub model repo (default)** | `AI4PD/REXzyme` | Downloads `training_dataset.json` and `validation_dataset.json` from the [model repo](https://huggingface.co/AI4PD/REXzyme) |
| **Other Hub dataset** | any `owner/name` | `load_dataset(dataset_id, …)` |
| **Local file or folder** | Path that exists on disk | `load_dataset("json", data_files=…)` |

Those Hub files are JSONL. Each line is `{"translation": {"Reaction": "<SMILES>", "Sequence": "<protein>"}}`. The encoder uses `Reaction`; the decoder uses `Sequence` (`Protein sequence` is accepted as a fallback).

- A **file** (`.json` / `.jsonl`) is loaded as JSON lines or array JSON.
- A **directory** infers splits named `train.jsonl`, `val.jsonl`, `test.jsonl`.
- Override splits with `--data_files split=path`.

Optional Hub flags: `--revision`, `--trust_remote_code`, `--split`, `--cache_dir`.

Outputs: a `DatasetDict` in `--output_dir` (`dataset_dict.json`) and
`tokenize_metadata.json`.

## Example training

[`../scripts/run_train.sh`](../scripts/run_train.sh) runs `tokenize.py` when
`TOKENIZED_DIR/dataset_dict.json` is missing (or when `FORCE_TOKENIZE=1`),
then [`train.py`](train.py).

Required: `TOKENIZER_AA`, `TOKENIZER_MOL`, `MODEL_NAME_OR_PATH`.

```bash
export TOKENIZER_AA=AI4PD/REXzyme/tokenizer_aa
export TOKENIZER_MOL=AI4PD/REXzyme/tokenizer_smiles
export MODEL_NAME_OR_PATH=AI4PD/REXzyme/checkpoint-90000   # fine-tune from paper checkpoint
export DATASET=AI4PD/REXzyme
bash scripts/run_train.sh
```

Local splits instead of the Hub:

```bash
export LOCAL_DATASET_PATH=/path/to/splits_dir
bash scripts/run_train.sh
```

`train.py` loads `AutoTokenizer` from the `tokenizer_aa` path recorded in
`tokenize_metadata.json`.
