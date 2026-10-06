#!/usr/bin/env bash
set -euo pipefail

# Run Rosetta metric extraction over structures.
# Expected downstream CSV columns:
# sequence_md5,rosetta_score,relative_hydrophobic_sasa,net_charge,average_sap,hydrophobic_sasa_to_ideal_ratio

PDB_DIR="${PDB_DIR:-results/alphafold/pdbs}"
OUTPUT_DIR="${OUTPUT_DIR:-results/rosetta}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ROSETTA_SCRIPT="${ROSETTA_SCRIPT:-job_scripts/get_rosetta_metrics.py}"

mkdir -p "${OUTPUT_DIR}"

for pdb_file in "${PDB_DIR}"/*.pdb; do
  [ -e "$pdb_file" ] || continue
  base="$(basename "${pdb_file}" .pdb)"
  ${PYTHON_BIN} "${ROSETTA_SCRIPT}" "${pdb_file}" > "${OUTPUT_DIR}/${base}_metrics.txt"
done

echo "Rosetta metric extraction completed. Convert outputs to standardized CSV."
