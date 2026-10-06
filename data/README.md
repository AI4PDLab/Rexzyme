# Paper data

Reaction SMILES used to generate the paper sequences, the sequences themselves, and Rhea / EC annotations.

## Reaction lists

One SMILES per line. Inference writes 25 sequences per line (see [`scripts/run_inference.sh`](../scripts/run_inference.sh)).

| File | Reactions | Training support |
|------|-----------|------------------|
| [`generation_most.txt`](generation_most.txt) | 20 | >10,000 sequences / reaction |
| [`generation_middle.txt`](generation_middle.txt) | 20 | 100–10,000 |
| [`generation_low.txt`](generation_low.txt) | 20 | <100 |
| [`generation_val_com.txt`](generation_val_com.txt) | 20 | 0, same EC in training |
| [`generation_val_dif.txt`](generation_val_dif.txt) | 20 | 0, same EC not in training |
| [`generation_heldout.txt`](generation_heldout.txt) | 1,256 (0–1255) | held-out Rhea reactions |

The last held-out reaction (`output_file_1255`) is the formate dehydrogenase reverse reaction (Rhea `15985`, EC `1.17.1.9`).

## Generated sequences

[`generated/`](generated/) holds the paper FASTAs (25 sequences per reaction). See [`generated/README.md`](generated/README.md).

## Rhea / EC / UniProt

[`extended_reaction_information/`](extended_reaction_information/) has one TSV per group (`smiles`, `rhea_id`, `ec`). The held-out folder also has `rhea_uniprot_sequence_check_best.tsv` for downloading ground-truth structures. See [`extended_reaction_information/README.md`](extended_reaction_information/README.md).

## Training and validation sets (Hugging Face)

The full training and validation JSON files are not stored in this GitHub repo
(they are several GB). They live next to the model on
[Hugging Face: AI4PD/REXzyme](https://huggingface.co/AI4PD/REXzyme)
(`training_dataset.json`, `validation_dataset.json`). Each line is JSONL:

```json
{"translation": {"Reaction": "<reaction SMILES>", "Sequence": "<protein sequence>"}}
```

```bash
python scripts/download_hf_assets.py
```

Eval scripts that need the training reactions default to that Hub repo. To
build a training FASTA for MMseqs:

```bash
python eval/train_identity/mmseq_file_preparation.py \
  --json_file AI4PD/REXzyme \
  --output training_dataset.fasta
```

Held-out TM-score tables from the paper (pLDDT, fingerprint distance, sequence
identity, embedding distance) are in
[`eval/tmscore_plots/`](../eval/tmscore_plots/).
