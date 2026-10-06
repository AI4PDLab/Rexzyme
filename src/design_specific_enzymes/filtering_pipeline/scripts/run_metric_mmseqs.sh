#!/usr/bin/env bash
set -euo pipefail

# Run MMseqs search against natural sequences.
# Expected downstream CSV columns: sequence_md5,identity_to_natural

QUERY_FASTA="${QUERY_FASTA:-results/dpo_generated.fasta}"
TARGET_FASTA="${TARGET_FASTA:-data/natural_sequences.fasta}"
OUTPUT_DIR="${OUTPUT_DIR:-results/mmseqs}"
TMP_DIR="${TMP_DIR:-${OUTPUT_DIR}/tmp}"

mkdir -p "${OUTPUT_DIR}" "${TMP_DIR}"

mmseqs easy-search \
  "${QUERY_FASTA}" \
  "${TARGET_FASTA}" \
  "${OUTPUT_DIR}/hits.tsv" \
  "${TMP_DIR}" \
  --format-output "query,target,pident,alnlen,evalue,bits"

echo "MMseqs completed. Convert hits.tsv to sequence_md5,identity_to_natural CSV."
