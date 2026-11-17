# Filtering Pipeline for Designed Enzymes

This folder defines a practical post-generation filtering workflow for
Rexzyme-designed sequences.

Use it after sequence generation (for example after fine-tuning and/or DPO).

## Goal

Start from a large generated set and progressively keep only candidates that are:

- biochemically plausible
- functionally aligned with target EC
- structurally confident
- diverse vs natural enzymes
- high quality by sequence and structure scoring models

## Protocol scope

This protocol starts from the FASTA file generated after DPO optimization
(for example, output from `seq_gen.py`) and ends with a final filtered CSV of
high-priority candidates.

## Recommended 11-step filtering order

1. **Remove duplicates**
   - Remove exact duplicate sequences first.

2. **Perplexity filter**
   - Compute conditional perplexity from the generation model.
   - Keep best 25% (lowest perplexity).

3. **CLEAN EC filter**
   - Run CLEAN predictor.
   - Keep sequences where predicted EC label matches the target EC (level can be
     strict 1-4 or relaxed as needed).

4. **AlphaFold structural confidence**
   - Fold sequences with AlphaFold.
   - Use the oligomeric state expected for your enzyme (for FDH it should be a
     dimer; adapt for other enzymes).
   - Keep sequences with mean pLDDT > 75.

5. **MMseqs identity vs natural sequences**
   - Remove sequences with 100% identity to natural enzymes.

6. **ipTM filter**
   - Keep best 25% by ipTM.

7. **Foldseek vs ground-truth structure**
   - Compare to a known reference structure (for example CtFDH for FDHs).
   - Keep best TM-score candidates (example: best 1000 sequences).

8. **ESM1v sequence score**
   - Score remaining sequences with ESM1v.
   - Keep best subset (example: best 500).

9. **ProteinMPNN score**
   - Score structures/sequences with ProteinMPNN.
   - Keep best subset (example: best 200).

10. **Interaction + catalytic sanity checks**
   - Use PLIP to compare interactions against a ground-truth structure.
   - Run MSA to confirm catalytic residues and key motifs are preserved.

11. **Rosetta metrics gate**
   - Compute Rosetta metrics (see
     [get_rosetta_metrics.py](https://github.com/Kortemme-Lab/protein_feature_analysis/blob/master/job_scripts/get_rosetta_metrics.py)).
   - Keep sequences within literature-recommended ranges and within the range of
     natural CO2-reducing enzymes.
   - Suggested checks:
     - best Rosetta score (lower is better)
     - relative hydrophobic SASA < 3
     - net charge not equal to 0 (to reduce aggregation risk)
     - average SAP < 0.5
     - hydrophobic SASA < 1.7 x ideal surface hydrophobic SASA (ideal sphere for
       same sequence length)
   - Filtering metrics reference:
     [Miao et al., 2022](https://www.biorxiv.org/content/10.1101/2022.12.21.521521v1)

## Suggested outputs per stage

- `step01_deduplicated.csv`
- `step02_perplexity_top25.csv`
- `step03_clean_ec_match.csv`
- `step04_plddt_gt75.csv`
- `step05_mmseqs_nonidentical.csv`
- `step06_iptm_top25.csv`
- `step07_foldseek_top1000.csv`
- `step08_esm1v_top500.csv`
- `step09_mpnn_top200.csv`
- `step10_plip_msa_checked.csv`
- `step11_rosetta_final.csv`

## External tools used in this pipeline

- CLEAN
- AlphaFold
- MMseqs2
- Foldseek
- ESM1v
- ProteinMPNN
- PLIP
- Rosetta / PyRosetta

This repository provides Rexzyme generation and evaluation utilities, while the
tools above are external dependencies that should be installed in dedicated
environments.

## Cluster execution notes

SLURM launchers for ESM1v, ProteinMPNN, and Rosetta are expected to stay
environment-specific. We recommend keeping those launchers local to each compute
environment, and keeping filtering thresholds/versioned decisions in this
repository.

## Configuration template

See [`filtering_config_template.yaml`](filtering_config_template.yaml) for a
parameterized version of this workflow (thresholds, top-k values, reference
paths, and oligomer settings).

## How to run the filtering scripts

There are two recommended operating modes:

- **Step-by-step mode**: use this when we want to inspect each stage manually,
  tune thresholds, or debug one metric.
- **Full-chain mode**: use this when we already have one merged scoring table and
  we want a default end-to-end pass.

Before either mode, we first build and maintain a single candidate table that
collects all metric outputs.

## Build one candidate table from seq_gen output

### 1) Initialize the table from FASTA

```bash
python src/design_specific_enzymes/filtering_pipeline/scripts/build_candidates_from_fasta.py \
  --input_fasta results/dpo_generated.fasta \
  --output_csv results/filtering/candidates_base.csv \
  --target_ec 1.17.1.9
```

This writes an initial table with:
- candidate id
- sequence
- sequence hash (`sequence_md5`)
- sequence length
- perplexity (parsed from FASTA header when present)
- optional target EC / reaction

### 2) Run each metric externally and merge its output

For each external metric tool, we recommend exporting a CSV keyed by
`sequence_md5` and then merging into the main table.

Merge helper:

```bash
python src/design_specific_enzymes/filtering_pipeline/scripts/merge_metric_table.py \
  --base_csv results/filtering/candidates_base.csv \
  --metric_csv results/filtering/clean_scores.csv \
  --output_csv results/filtering/candidates_with_clean.csv \
  --base_key sequence_md5 \
  --metric_key sequence_md5 \
  --columns predicted_ec
```

Repeat this merge step for every metric table (AlphaFold, MMseqs, Foldseek,
ESM1v, ProteinMPNN, Rosetta, etc.) until one merged file contains all columns.

## How to run in silico metrics (and what to merge)

The metric computations are external; this repo provides the filtering logic.
For each metric, we run the external tool and then merge its output columns into
the candidate table.

1) **CLEAN**
- Run CLEAN inference and parse predicted EC labels.
- Merge column: `predicted_ec`.

2) **AlphaFold**
- Fold with the expected oligomeric state for the enzyme family.
- Merge columns: `plddt`, `iptm`.

3) **MMseqs**
- Search against natural/reference sequences.
- Merge column: `identity_to_natural`.

4) **Foldseek**
- Compare to ground-truth structures.
- Merge column: `tmscore`.

5) **ESM1v**
- Score sequence plausibility.
- Merge column: `esm1v_score`.

6) **ProteinMPNN**
- Score sequence-structure compatibility.
- Merge column: `proteinmpnn_score`.

7) **PLIP + MSA**
- Manual/automated interaction and motif checks.
- Optionally merge flags/annotations used for manual gating.

8) **Rosetta metrics**
- Compute Rosetta-based quality values.
- Merge columns:
  - `rosetta_score`
  - `relative_hydrophobic_sasa`
  - `net_charge`
  - `average_sap`
  - `hydrophobic_sasa_to_ideal_ratio`

### Pre-analysis metric scripts

To run all metric stages before filtering:

```bash
bash src/design_specific_enzymes/filtering_pipeline/scripts/run_all_metrics.sh
```

To run one metric stage at a time:

```bash
bash src/design_specific_enzymes/filtering_pipeline/scripts/run_metric_perplexity.sh
bash src/design_specific_enzymes/filtering_pipeline/scripts/run_metric_clean.sh
bash src/design_specific_enzymes/filtering_pipeline/scripts/run_metric_alphafold.sh
bash src/design_specific_enzymes/filtering_pipeline/scripts/run_metric_mmseqs.sh
bash src/design_specific_enzymes/filtering_pipeline/scripts/run_metric_foldseek.sh
bash src/design_specific_enzymes/filtering_pipeline/scripts/run_metric_esm1v.sh
bash src/design_specific_enzymes/filtering_pipeline/scripts/run_metric_proteinmpnn.sh
bash src/design_specific_enzymes/filtering_pipeline/scripts/run_metric_plip.sh
bash src/design_specific_enzymes/filtering_pipeline/scripts/run_metric_rosetta.sh
```

Each script is a template around the corresponding external tool command. We can
adjust paths, environment activation, and cluster directives per infrastructure.

In step-by-step mode, we usually run one stage script at a time and check row
counts after every stage.

### Step-by-step mode

1) Deduplicate FASTA sequences before scoring. This reads
`results/dpo_generated.fasta` (the default output of the DPO launcher) and
writes both `results/step01_deduplicated.fasta` and the candidate table
`results/step01_deduplicated.csv` used by the next steps:

```bash
bash src/design_specific_enzymes/filtering_pipeline/scripts/run_deduplicate.sh
```

The perplexity column is read from the `ppl_<value>` tag in the FASTA headers
written by `dpo_optimize_and_generate.py`. FASTA files from `src/inference.py`
have no such tag; in that case the perplexity filter stops with an error until
perplexity values are merged into the table (`merge_metric_table.py`).

2) Keep the best fraction by perplexity (default 25%):

```bash
bash src/design_specific_enzymes/filtering_pipeline/scripts/run_perplexity_filter.sh
```

3) Keep CLEAN EC matches to the target EC:

```bash
bash src/design_specific_enzymes/filtering_pipeline/scripts/run_clean_filter.sh
```

4) Apply AlphaFold-derived pLDDT and ipTM filters:

```bash
bash src/design_specific_enzymes/filtering_pipeline/scripts/run_alphafold_filter.sh
```

5) Remove natural-sequence identity hits at 100%:

```bash
bash src/design_specific_enzymes/filtering_pipeline/scripts/run_mmseqs_filter.sh
```

6) Keep top Foldseek hits by TM-score:

```bash
bash src/design_specific_enzymes/filtering_pipeline/scripts/run_foldseek_filter.sh
```

7) Keep top ESM1v candidates:

```bash
bash src/design_specific_enzymes/filtering_pipeline/scripts/run_esm1v_filter.sh
```

8) Keep top ProteinMPNN candidates:

```bash
bash src/design_specific_enzymes/filtering_pipeline/scripts/run_proteinmpnn_filter.sh
```

9) After manual PLIP/MSA review, apply Rosetta gates:

```bash
bash src/design_specific_enzymes/filtering_pipeline/scripts/run_rosetta_filter.sh
```

### Full-chain mode

Use this when a single CSV already contains all required score columns
(`perplexity`, `predicted_ec`, `target_ec`, `plddt`, `identity_to_natural`,
`iptm`, `tmscore`, `esm1v_score`, `proteinmpnn_score`, Rosetta metrics):

```bash
bash src/design_specific_enzymes/filtering_pipeline/scripts/run_filtering_pipeline.sh
```

### When to use low-level Python filters directly

Use Python filter utilities directly when we need custom logic not covered by
the shell templates (for example a different threshold, another score column, or
alternative top-k cutoffs).

Example: custom pLDDT threshold (`>= 80`):

```bash
python src/design_specific_enzymes/filtering_pipeline/scripts/filter_by_threshold.py \
  --input_csv results/step03_clean_ec_match.csv \
  --output_csv results/step04_plddt_gt80.csv \
  --column plddt \
  --op ge \
  --value 80
```

Example: custom top 10% by ESM1v:

```bash
python src/design_specific_enzymes/filtering_pipeline/scripts/filter_top_fraction.py \
  --input_csv results/step07_foldseek_top1000.csv \
  --output_csv results/step08_esm1v_top10pct.csv \
  --column esm1v_score \
  --fraction 0.10 \
  --direction higher_is_better
```
