#!/usr/bin/env bash
set -euo pipefail

# Run CLEAN on one FASTA and export predicted EC labels.
#
# Requires:
# - CLEAN cloned and installed
# - CLEAN app folder structure

INPUT_FASTA="${INPUT_FASTA:-results/dpo_generated.fasta}"
TARGET_EC="${TARGET_EC:-}"
REXZYME_ROOT="${REXZYME_ROOT:-$(pwd)}"
CLEAN_APP_DIR="${CLEAN_APP_DIR:-/path/to/CLEAN/app}"
OUTPUT_CSV="${OUTPUT_CSV:-results/filtering/clean_scores.csv}"

FASTA_BASENAME="$(basename "${INPUT_FASTA}")"
FASTA_STEM="${FASTA_BASENAME%.*}"

mkdir -p "${CLEAN_APP_DIR}/data/inputs"
cp "${INPUT_FASTA}" "${CLEAN_APP_DIR}/data/inputs/${FASTA_BASENAME}"

python "${REXZYME_ROOT}/eval/clean/CLEAN_infer_fasta.py" -d "${FASTA_STEM}"

python "${REXZYME_ROOT}/design_specific_enzymes/filtering_pipeline/scripts/parse_clean_predictions.py" \
  --clean_output "${CLEAN_APP_DIR}/results/inputs/${FASTA_STEM}.csv" \
  --input_fasta "${INPUT_FASTA}" \
  --output_csv "${OUTPUT_CSV}" \
  --target_ec "${TARGET_EC}"
