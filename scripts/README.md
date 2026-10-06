# Scripts

Thin wrappers around [`src/`](../src/). Run them from the repository root.

The paper **model**, **tokenizers**, and **train/validation JSON** are on
[Hugging Face: AI4PD/REXzyme](https://huggingface.co/AI4PD/REXzyme). Inference
and training load those Hub ids directly (`checkpoint-90000`, `tokenizer_aa`,
`tokenizer_smiles`). Use [`download_hf_assets.py`](download_hf_assets.py) only
when you need the large JSON files on disk.

## Generate sequences (paper settings)

[`run_inference.sh`](run_inference.sh) calls [`src/inference.py`](../src/inference.py) with `top_k=100`, `top_p=1.0`, `repetition_penalty=1.0`, and 25 sequences per reaction.

```bash
export INPUT_FILE=data/generation_most.txt
export OUTPUT_FOLDER=generation_output
bash scripts/run_inference.sh
```

| Variable | Default |
|----------|---------|
| `INPUT_FILE` | `data/generation_most.txt` |
| `OUTPUT_FOLDER` | `generation_output` |
| `MODEL_PATH` | `AI4PD/REXzyme/checkpoint-90000` |
| `TOKENIZER_AA` | `AI4PD/REXzyme/tokenizer_aa` |
| `TOKENIZER_MOL` | `AI4PD/REXzyme/tokenizer_smiles` |
| `TOP_K` | `100` |
| `TOP_P` | `1.0` |
| `REPETITION_PENALTY` | `1.0` |
| `NUM_RETURN_SEQUENCES` | `25` |
| `SEED` | `0` |

## Download train / validation JSON

[`download_hf_assets.py`](download_hf_assets.py) caches `training_dataset.json`
and `validation_dataset.json` from [AI4PD/REXzyme](https://huggingface.co/AI4PD/REXzyme).
Each file is JSONL: `{"translation": {"Reaction": "<SMILES>", "Sequence": "<protein>"}}`.

```bash
python scripts/download_hf_assets.py
# optional: python scripts/download_hf_assets.py --files training_dataset.json
```

## Example training

[`run_train.sh`](run_train.sh) tokenizes (if needed) and runs [`src/train.py`](../src/train.py). See [`src/README.md`](../src/README.md).

| Variable | Default |
|----------|---------|
| `TOKENIZER_AA` | required |
| `TOKENIZER_MOL` | required |
| `MODEL_NAME_OR_PATH` | required |
| `DATASET` | `AI4PD/REXzyme` (downloads train/val JSON) |
| `LOCAL_DATASET_PATH` | unset (use `DATASET`) |
| `TOKENIZED_DIR` | `data/tokenized` |
| `OUTPUT_DIR` | `results/training_output` |
| `FORCE_TOKENIZE` | `0` — set to `1` to rebuild tokenized data even if it exists |
| `HF_REVISION` | unset — pass a Hub branch/tag/commit to pin the dataset version |
| `TRUST_REMOTE_CODE` | `0` — set to `1` for Hub datasets that require it |
| `TRAIN_EXTRA_ARGS` | unset — extra flags appended verbatim to `src/train.py` (e.g. `"--num_train_epochs 5"`) |
