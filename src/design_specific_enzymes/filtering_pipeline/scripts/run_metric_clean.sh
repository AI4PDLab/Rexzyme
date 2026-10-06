#!/usr/bin/env bash
set -euo pipefail

# Run CLEAN on one FASTA and export predicted EC labels.
#
# Requires:
# - CLEAN cloned and installed
# - CLEAN app folder structure
#
# CLEAN_infer_fasta.py uses paths relative to CLEAN/app (data/inputs,
# data/pretrained, results/), so it is run from CLEAN_APP_DIR.

INPUT_FASTA="${INPUT_FASTA:-results/dpo_generated.fasta}"
TARGET_EC="${TARGET_EC:-}"
REXZYME_ROOT="${REXZYME_ROOT:-$(pwd)}"
CLEAN_APP_DIR="${CLEAN_APP_DIR:-/path/to/CLEAN/app}"
OUTPUT_CSV="${OUTPUT_CSV:-results/filtering/clean_scores.csv}"

if [ ! -d "${CLEAN_APP_DIR}" ]; then
  echo "ERROR: CLEAN_APP_DIR does not exist: ${CLEAN_APP_DIR} (set it to CLEAN/app)" >&2
  exit 1
fi

# Resolve paths that are relative to the Rexzyme root before changing directory.
INPUT_FASTA="$(realpath "${INPUT_FASTA}")"
mkdir -p "$(dirname "${OUTPUT_CSV}")"
OUTPUT_CSV="$(realpath -m "${OUTPUT_CSV}")"
CLEAN_APP_DIR="$(realpath "${CLEAN_APP_DIR}")"

FASTA_BASENAME="$(basename "${INPUT_FASTA}")"
FASTA_STEM="${FASTA_BASENAME%.*}"

mkdir -p "${CLEAN_APP_DIR}/data/inputs"
cp "${INPUT_FASTA}" "${CLEAN_APP_DIR}/data/inputs/${FASTA_BASENAME}"

(cd "${CLEAN_APP_DIR}" && python "${REXZYME_ROOT}/eval/clean/CLEAN_infer_fasta.py" -d "${FASTA_STEM}")

# Depending on the CLEAN version the prediction file is <stem>.csv or <stem>_maxsep.csv.
CLEAN_OUTPUT=""
for candidate in "${CLEAN_APP_DIR}/results/inputs/${FASTA_STEM}.csv" \
                 "${CLEAN_APP_DIR}/results/inputs/${FASTA_STEM}_maxsep.csv"; do
  if [ -f "${candidate}" ]; then
    CLEAN_OUTPUT="${candidate}"
    break
  fi
done
if [ -z "${CLEAN_OUTPUT}" ]; then
  echo "ERROR: no CLEAN output found in ${CLEAN_APP_DIR}/results/inputs/ for ${FASTA_STEM}" >&2
  exit 1
fi

python "${REXZYME_ROOT}/src/design_specific_enzymes/filtering_pipeline/scripts/parse_clean_predictions.py" \
  --clean_output "${CLEAN_OUTPUT}" \
  --input_fasta "${INPUT_FASTA}" \
  --output_csv "${OUTPUT_CSV}" \
  --target_ec "${TARGET_EC}"
