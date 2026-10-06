# Held-out structure superposition

After folding the sequences in [`data/generated/generation_heldout/`](../../data/generated/generation_heldout/), compare each design to natural ground-truth structures with Foldseek.

The held-out set is reactions **0–1255** (1,256 reactions), including the formate dehydrogenase reverse reaction (`output_file_1255`, Rhea `15985`, EC `1.17.1.9`).

Foldseek is an external dependency (not installed by pip). Put `foldseek` on `PATH` or set `FOLDSEEK`.

## 1. Download ground-truth structures

```bash
python eval/gt_superpose/download_gt_pdbs.py \
  --smiles-map data/generated/generation_heldout/smiles_map.json \
  --outdir gt_structures
```

Defaults:

- SMILES map: [`data/generated/generation_heldout/smiles_map.json`](../../data/generated/generation_heldout/smiles_map.json)
- UniProt / Rhea table: [`data/extended_reaction_information/generation_heldout/rhea_uniprot_sequence_check_best.tsv`](../../data/extended_reaction_information/generation_heldout/rhea_uniprot_sequence_check_best.tsv)

Creates `gt_structures/gt_output_file_<N>/` with up to 5 experimental PDBs (or AlphaFold models). RCSB downloads try `.pdb` first and fall back to `.cif` when the entry is mmCIF-only. For `best_criterion == PDB`, curated UniProt accessions are queried first (so 1255 prefers 2FSS / 6T8Y). To download only that reaction:

```bash
python eval/gt_superpose/download_gt_pdbs.py --outdir gt_structures --start 1255 --end 1255
```

A folder with a `DONE` marker is skipped. Use `--force` to re-download.

## 2. Superpose designs onto ground truth

Fold designs first ([`eval/plddt/run-esm.py`](../plddt/run-esm.py)). That script writes a flat folder of `top_*.pdb` files; this step expects one folder per reaction:

```
designs/output_file_<N>/pdbs/*.pdb
```

(`designs/output_file_<N>/*.pdb` is also accepted.)

```bash
bash eval/gt_superpose/superpose_designs_to_gt.sh \
  --designs /path/to/designs \
  --gt /path/to/gt_structures \
  --out /path/to/superposition_ground_truth_designs \
  --start 0 \
  --end 1255
```

The default range is `0 1255`. Outputs per reaction plus:

- `all_tmscores_unfiltered.tsv` — every Foldseek hit
- `all_tmscores.tsv` — reliable hits only

If the aggregate files already exist, new rows are appended. Delete them (or set a new `OUT`) before re-running a range that was already processed, or you will get duplicate rows. To add only reaction 1255 to an existing 0–1254 run:

```bash
bash eval/gt_superpose/superpose_designs_to_gt.sh 1255 1255
```

## 3. Re-apply the reliability filter

```bash
python eval/gt_superpose/apply_reliability_filter.py \
  --root /path/to/superposition_ground_truth_designs \
  --input /path/to/superposition_ground_truth_designs/all_tmscores_unfiltered.tsv \
  --output /path/to/superposition_ground_truth_designs/all_tmscores.tsv
```

A pair is kept only if `tcov >= 0.5`, `tlen >= 50`, and `qcov >= 0.5`.

## 4. Plot TM-score against other metrics

[`../tmscore_plots/README.md`](../tmscore_plots/README.md) plots the filtered
`all_tmscores.tsv` against pLDDT, fingerprint distance, sequence identity, and
encoder embedding distance. Paper tables for those plots (reactions 0–1255)
are already in [`../tmscore_plots/`](../tmscore_plots/).
