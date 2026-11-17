#!/usr/bin/env bash
set -euo pipefail

# Run ProteinMPNN scoring on predicted structures.
# Expected downstream CSV columns: sequence_md5,proteinmpnn_score

PDB_DIR="${PDB_DIR:-results/alphafold/pdbs}"
OUTPUT_DIR="${OUTPUT_DIR:-results/proteinmpnn}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MPNN_SCRIPT="${MPNN_SCRIPT:-protein_mpnn_run.py}"
CHAIN_IDS="${CHAIN_IDS:-A}"

mkdir -p "${OUTPUT_DIR}"

for pdb_file in "${PDB_DIR}"/*.pdb; do
  [ -e "$pdb_file" ] || continue
  ${PYTHON_BIN} "${MPNN_SCRIPT}" \
    --pdb_path "${pdb_file}" \
    --pdb_path_chains "${CHAIN_IDS}" \
    --score_only 1 \
    --save_score 1 \
    --out_folder "${OUTPUT_DIR}" \
    --batch_size 1
done

echo "ProteinMPNN completed. Parse outputs to sequence_md5,proteinmpnn_score CSV."
