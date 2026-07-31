# Rexzyme
### _Code for the paper:_ 
### [Rexzyme: A Translation Machine for the design of enzymes tailored to specific reactions] _To Do: Link_

Generate enzyme sequences from reaction SMILES using an encoder–decoder Protein Language Model.

## Requirements

- Python 3.10+
- Core Python packages are listed in [`requirements.txt`](requirements.txt): `torch`, `transformers`, `datasets`, `numpy`.
- Evaluation extras are listed in [`requirements-eval.txt`](requirements-eval.txt): `pandas`, `matplotlib`, `biopython`.
- MMseqs is an external system dependency for the evaluation scripts. It is not installed by pip; eval scripts expect `mmseqs` on `PATH` or an explicit MMseqs executable path where the script exposes one.

## Pretrained Models

The pretrained models presented in the paper are available on [Hugging Face](https://huggingface.co/AI4PD/REXzyme)

## Running inference

**Input:** a text file with one SMILES per line (or a JSON/JSONL file with translation entries).

**Example — model and tokenizers on Hugging Face:**

```bash
python src/inference.py \
  --model_path AI4PD/REXzyme/checkpoint-90000 \
  --tokenizer_aa AI4PD/REXzyme/tokenize_aa \
  --tokenizer_mol AI4PD/REXzyme/tokenize_smiles \
  --input_file data/guide_val_com.txt \
  --output_folder fastas \
  --top_k 15 \
  --seed 0
```


Outputs: one FASTA per input SMILES (25 sequences per molecule by default), plus `inference_metadata.json` (per-FASTA SMILES, perplexities, optional ground-truth sequences; run-level greedy perplexity stats) and `generation_parameters.json` in `--output_folder`.

## Citation

If you find this work useful, please cite:

_To Do_
