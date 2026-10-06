# Structure confidence (pLDDT)

Fold generated sequences with ESMFold and plot mean pLDDT by reaction group.
The five small `generation_*` sets (not the held-out set) are used for the
group plot.

ESMFold is loaded from Hugging Face (`facebook/esmfold_v1` by default).

## 1. Fold sequences

[`run-esm.py`](run-esm.py) reads every `*.fasta` in `--fasta_path` and writes
PDBs to `--output_path`:

```bash
python eval/plddt/run-esm.py \
  --fasta_path data/generated/generation_most \
  --output_path fold/generation_most
```

`--fasta_path` / `--input` is a FASTA file or a folder of FASTAs.
`--output_path` / `--output` is the directory for PDB files.

For held-out superposition, put PDBs in `designs/output_file_<N>/pdbs/`
(see [`../gt_superpose/README.md`](../gt_superpose/README.md)).

## 2. Plot pLDDT by group

[`plot_plddt_by_group.py`](plot_plddt_by_group.py) expects one folder per group
under `--root`:

```
fold/generation_most/*.pdb
fold/generation_middle/*.pdb
fold/generation_low/*.pdb
fold/generation_val_com/*.pdb
fold/generation_val_dif/*.pdb
```

```bash
python eval/plddt/plot_plddt_by_group.py \
  --fold-dir fold \
  --output plddt_by_training_group
```

`--fold-dir` / `--input` is the directory that contains `generation_most`,
`generation_middle`, and the other group folders. `--output` is the plot
and TSV prefix.

[`plot_plddt.py`](plot_plddt.py) is a per-folder scatter from a single ESMFold
output directory:

```bash
python eval/plddt/plot_plddt.py \
  --results_folder fold/generation_most \
  --output fold/generation_most/average_scatter_plot.png \
  --top_k 100
```


Held-out TM-score versus pLDDT uses a different table:
[`../tmscore_plots/tmscore_vs_plddt.tsv`](../tmscore_plots/tmscore_vs_plddt.tsv).


