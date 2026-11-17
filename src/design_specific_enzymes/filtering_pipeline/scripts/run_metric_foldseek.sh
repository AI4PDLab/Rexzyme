#!/usr/bin/env bash
set -euo pipefail

# Run Foldseek structure search against a ground-truth structure set.
# Expected downstream CSV columns: sequence_md5,tmscore

QUERY_PDB_DIR="${QUERY_PDB_DIR:-results/alphafold/pdbs}"
TARGET_PDB_DIR="${TARGET_PDB_DIR:-data/ground_truth_structures}"
OUTPUT_DIR="${OUTPUT_DIR:-results/foldseek}"
TMP_DIR="${TMP_DIR:-${OUTPUT_DIR}/tmp}"

mkdir -p "${OUTPUT_DIR}" "${TMP_DIR}"

foldseek easy-search \
  "${QUERY_PDB_DIR}" \
  "${TARGET_PDB_DIR}" \
  "${OUTPUT_DIR}/hits.tsv" \
  "${TMP_DIR}" \
  --format-output "query,target,alntmscore,qtmscore,ttmscore,evalue,bits"

echo "Foldseek completed. Convert hits.tsv to sequence_md5,tmscore CSV."
