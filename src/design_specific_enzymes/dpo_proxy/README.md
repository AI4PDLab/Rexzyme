# DPO Optimization by User Proxy

`dpo_optimize_and_generate.py` is a port of the original `DPO_pLM.py` +
`seq_gen.py` pipeline to the current Rexzyme repository structure.

It works with encoder-decoder checkpoints (finetuned Rexzyme or compatible
seq2seq models) and lets you optimize with:

- `--mode weighted`
- `--mode ranked`

using any numeric proxy column from a table (for example `Kcat`, `Km`,
`Kcat/Km`, `plddt`, `tmscore`, or any custom score you supply).

## Loss functions (matching original DPO_pLM.py)

| Mode | Loss |
|------|------|
| `weighted` | `F.cross_entropy(β · π_ll, softmax(scores))` |
| `ranked` | Sort by score weight, then `F.mse_loss(β · π_ll, ones)` |

Log-likelihood is computed per sequence using `model.loss` (averaged NLL
negated), iterating one sequence at a time — same as the original script.
The reference model is loaded but not used in the loss, matching the original
`DPO_pLM.py` behaviour.

## Input table

Supported formats: `.csv`, `.tsv` / `.txt` (tab-separated), `.xlsx` / `.xls`

Template: [`example_proxy_input.csv`](example_proxy_input.csv)

Required columns:

- one sequence column (default: `Seq`)
- one numeric proxy column (`--proxy_column`)
- reaction SMILES: either a single value via `--reaction`, or a per-row
  column via `--reaction_column`

## Typical usage

```bash
python src/design_specific_enzymes/dpo_proxy/dpo_optimize_and_generate.py \
  --model_path /path/to/finetuned_model \
  --output_model_dir results/dpo_model \
  --tokenizer_aa AI4PD/REXzyme/tokenizer_aa \
  --tokenizer_mol AI4PD/REXzyme/tokenizer_smiles \
  --activity_file /path/to/FDH_activity.csv \
  --sequence_column Seq \
  --proxy_column "Kcat/Km" \
  --reaction "SMILES_REACTION_HERE" \
  --mode ranked \
  --objective maximize \
  --generate_after_dpo \
  --generation_reaction "SMILES_REACTION_HERE" \
  --fasta_output results/fdh_dpo_ranked.fasta
```

Or use the launcher script (all parameters are overridable via env vars):

```bash
MODEL_PATH=/path/to/finetuned_model \
ACTIVITY_FILE=/path/to/activity.csv \
REACTION="SMILES_REACTION_HERE" \
PROXY_COLUMN="Kcat/Km" \
MODE=ranked \
bash src/design_specific_enzymes/dpo_proxy/run_dpo_and_generate.sh
```

## Default hyperparameters

Defaults match the original `DPO_pLM.py` CONFIG and `seq_gen.py`:

| Parameter | Default | Source |
|-----------|---------|--------|
| `--beta` | 0.1 | DPO_pLM.py |
| `--learning_rate` | 1e-7 | DPO_pLM.py |
| `--batch_size` | 8 | DPO_pLM.py |
| `--num_epochs` | 10 | DPO_pLM.py |
| `--seed` | 1998 | DPO_pLM.py |
| `--generation_rounds` | 500 | seq_gen.py |
| `--num_return_sequences` | 20 | seq_gen.py |
| `--top_k` | 18 | seq_gen.py |
| `--repetition_penalty` | 1.1 | seq_gen.py |

## Choosing optimization direction

- `--objective maximize`: larger is better (`Kcat`, `Kcat/Km`, `plddt`,
  `tmscore`)
- `--objective minimize`: smaller is better (`Km`, error-like metrics)

## Notes

- Best checkpoint (lowest eval loss) is saved to `--output_model_dir`.
- Tokenizer paths support Hub subfolder notation:
  `AI4PD/REXzyme/tokenizer_aa` resolves to repo `AI4PD/REXzyme`,
  subfolder `tokenizer_aa`.
- No `"translation"` prefix is prepended to reaction SMILES — the current
  model checkpoint was not trained with it.
