# Extended reaction information

The SMILES lists in [`data/generation_*.txt`](../) plus Rhea ID and EC label.

Each folder matches one generation group. The TSV has one row per reaction, in the same order as the original file:

| Column | Description |
|--------|-------------|
| `smiles` | Reaction SMILES (same as `data/<group>.txt`) |
| `rhea_id` | Rhea identifier(s) |
| `ec` | EC label(s) |

When one SMILES maps to several Rhea entries, IDs and EC labels are joined with `;`. `Not Found` means the mapping file had no EC for that entry.

[`generation_heldout/rhea_uniprot_sequence_check_best.tsv`](generation_heldout/rhea_uniprot_sequence_check_best.tsv) adds UniProt accessions used by [`eval/gt_superpose/download_gt_pdbs.py`](../../eval/gt_superpose/download_gt_pdbs.py) to download ground-truth structures.

| Folder | Source reactions | Rows |
|--------|------------------|------|
| [`generation_most/`](generation_most/) | [`../generation_most.txt`](../generation_most.txt) | 20 |
| [`generation_middle/`](generation_middle/) | [`../generation_middle.txt`](../generation_middle.txt) | 20 |
| [`generation_low/`](generation_low/) | [`../generation_low.txt`](../generation_low.txt) | 20 |
| [`generation_val_com/`](generation_val_com/) | [`../generation_val_com.txt`](../generation_val_com.txt) | 20 |
| [`generation_val_dif/`](generation_val_dif/) | [`../generation_val_dif.txt`](../generation_val_dif.txt) | 20 |
| [`generation_heldout/`](generation_heldout/) | [`../generation_heldout.txt`](../generation_heldout.txt) | 1,256 (0–1255) |

The last held-out row (`output_file_1255`) is the formate dehydrogenase reverse
reaction (Rhea `15985`, EC `1.17.1.9`).
