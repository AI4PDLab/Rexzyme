#!/usr/bin/env bash
set -euo pipefail

# Template: run AlphaFold externally, then filter table by pLDDT and ipTM.
# Assumes plddt and iptm columns are already merged into the CSV table.

INPUT_CSV="${INPUT_CSV:-results/step03_clean_ec_match.csv}"
PLDDT_CSV="${PLDDT_CSV:-results/step04_plddt_gt75.csv}"
IPTM_CSV="${IPTM_CSV:-results/step06_iptm_top25.csv}"

python design_specific_enzymes/filtering_pipeline/scripts/filter_by_threshold.py \
  --input_csv "${INPUT_CSV}" \
  --output_csv "${PLDDT_CSV}" \
  --column plddt \
  --op gt \
  --value 75

python design_specific_enzymes/filtering_pipeline/scripts/filter_top_fraction.py \
  --input_csv "${PLDDT_CSV}" \
  --output_csv "${IPTM_CSV}" \
  --column iptm \
  --fraction 0.25 \
  --direction higher_is_better
