#!/usr/bin/env bash
set -euo pipefail

# Template: compute Rosetta metrics externally and merge into CSV, then apply
# Rosetta gates.

INPUT_CSV="${INPUT_CSV:-results/step10_plip_msa_checked.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-results/step11_rosetta_final.csv}"

python design_specific_enzymes/filtering_pipeline/scripts/filter_rosetta_gates.py \
  --input_csv "${INPUT_CSV}" \
  --output_csv "${OUTPUT_CSV}" \
  --rosetta_score_column rosetta_score \
  --relative_hydrophobic_sasa_column relative_hydrophobic_sasa \
  --net_charge_column net_charge \
  --average_sap_column average_sap \
  --hydrophobic_ratio_column hydrophobic_sasa_to_ideal_ratio \
  --relative_hydrophobic_sasa_max 3.0 \
  --average_sap_max 0.5 \
  --hydrophobic_ratio_max 1.7 \
  --require_nonzero_net_charge
