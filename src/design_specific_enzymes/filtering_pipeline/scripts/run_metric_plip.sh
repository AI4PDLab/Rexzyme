#!/usr/bin/env bash
set -euo pipefail

# Run PLIP interaction extraction on structures.
# Expected downstream outputs: interaction reports and optional pass/fail flags.

PDB_DIR="${PDB_DIR:-results/alphafold/pdbs}"
OUTPUT_DIR="${OUTPUT_DIR:-results/plip}"
PLIP_CMD="${PLIP_CMD:-plipcmd}"

mkdir -p "${OUTPUT_DIR}"

for pdb_file in "${PDB_DIR}"/*.pdb; do
  [ -e "$pdb_file" ] || continue
  base="$(basename "${pdb_file}" .pdb)"
  ${PLIP_CMD} -f "${pdb_file}" -o "${OUTPUT_DIR}/${base}"
done

echo "PLIP completed. Use reports for interaction comparison and MSA checks."
