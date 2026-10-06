# Generated sequences

Sequences produced by Rexzyme for the paper reaction groups (25 sequences per reaction).
Regenerate them with [`../../scripts/run_inference.sh`](../../scripts/run_inference.sh)
(defaults: [AI4PD/REXzyme](https://huggingface.co/AI4PD/REXzyme) `checkpoint-90000`,
`tokenizer_aa`, `tokenizer_smiles`).

`output_topk100_file_N.fasta` contains the sequences for reaction `N` (0-based) in the matching file under [`data/`](../). `smiles_map.json` in each folder records the same FASTA-to-SMILES mapping.

| Folder | Reaction file | Sequences |
|--------|---------------|-----------|
| [`generation_most/`](generation_most/) | [`../generation_most.txt`](../generation_most.txt) | 20 × 25 = 500 |
| [`generation_middle/`](generation_middle/) | [`../generation_middle.txt`](../generation_middle.txt) | 20 × 25 = 500 |
| [`generation_low/`](generation_low/) | [`../generation_low.txt`](../generation_low.txt) | 20 × 25 = 500 |
| [`generation_val_com/`](generation_val_com/) | [`../generation_val_com.txt`](../generation_val_com.txt) | 20 × 25 = 500 |
| [`generation_val_dif/`](generation_val_dif/) | [`../generation_val_dif.txt`](../generation_val_dif.txt) | 20 × 25 = 500 |
| [`generation_heldout/`](generation_heldout/) | [`../generation_heldout.txt`](../generation_heldout.txt) | 1,256 × 25 = 31,400 |

These folders use the same `generation_*` names expected by the group-level scripts in [`eval/`](../../eval/).

[`generation_heldout/`](generation_heldout/) is the generalisation set: Rhea reactions not present in the training data (indices **0–1255**). Sequences are in a single FASTA, `all_sequences_with_filenames.fasta`. Headers are `output_file_N_K_ppl_...` (reaction `N`, sequence `K`). `smiles_map.json` maps `output_file_N.fasta` to the corresponding SMILES.

Paper TM-score analyses of this set: [`eval/tmscore_plots/`](../../eval/tmscore_plots/).
