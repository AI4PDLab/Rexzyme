#!/usr/bin/env bash
set -euo pipefail

# End-to-end filter pass on a merged scoring table.
#
# The input table should already contain columns:
# perplexity,predicted_ec,target_ec,plddt,identity_to_natural,iptm,tmscore,
# esm1v_score,proteinmpnn_score,rosetta_score,relative_hydrophobic_sasa,
# net_charge,average_sap,hydrophobic_sasa_to_ideal_ratio

IN="${IN:-results/merged_scores.csv}"
OUT_DIR="${OUT_DIR:-results/filtering}"
mkdir -p "${OUT_DIR}"

python src/design_specific_enzymes/filtering_pipeline/scripts/filter_top_fraction.py \
  --input_csv "${IN}" \
  --output_csv "${OUT_DIR}/step02_perplexity_top25.csv" \
  --column perplexity \
  --fraction 0.25 \
  --direction lower_is_better

python src/design_specific_enzymes/filtering_pipeline/scripts/filter_clean_ec_match.py \
  --input_csv "${OUT_DIR}/step02_perplexity_top25.csv" \
  --output_csv "${OUT_DIR}/step03_clean_ec_match.csv" \
  --predicted_ec_column predicted_ec \
  --target_ec_column target_ec \
  --level 4

python src/design_specific_enzymes/filtering_pipeline/scripts/filter_by_threshold.py \
  --input_csv "${OUT_DIR}/step03_clean_ec_match.csv" \
  --output_csv "${OUT_DIR}/step04_plddt_gt75.csv" \
  --column plddt \
  --op gt \
  --value 75

python src/design_specific_enzymes/filtering_pipeline/scripts/filter_mmseqs_identity.py \
  --input_csv "${OUT_DIR}/step04_plddt_gt75.csv" \
  --output_csv "${OUT_DIR}/step05_mmseqs_nonidentical.csv" \
  --identity_column identity_to_natural \
  --max_identity 0.9999

python src/design_specific_enzymes/filtering_pipeline/scripts/filter_top_fraction.py \
  --input_csv "${OUT_DIR}/step05_mmseqs_nonidentical.csv" \
  --output_csv "${OUT_DIR}/step06_iptm_top25.csv" \
  --column iptm \
  --fraction 0.25 \
  --direction higher_is_better

python src/design_specific_enzymes/filtering_pipeline/scripts/filter_top_n.py \
  --input_csv "${OUT_DIR}/step06_iptm_top25.csv" \
  --output_csv "${OUT_DIR}/step07_foldseek_top1000.csv" \
  --column tmscore \
  --top_n 1000 \
  --direction higher_is_better

python src/design_specific_enzymes/filtering_pipeline/scripts/filter_top_n.py \
  --input_csv "${OUT_DIR}/step07_foldseek_top1000.csv" \
  --output_csv "${OUT_DIR}/step08_esm1v_top500.csv" \
  --column esm1v_score \
  --top_n 500 \
  --direction higher_is_better

python src/design_specific_enzymes/filtering_pipeline/scripts/filter_top_n.py \
  --input_csv "${OUT_DIR}/step08_esm1v_top500.csv" \
  --output_csv "${OUT_DIR}/step09_mpnn_top200.csv" \
  --column proteinmpnn_score \
  --top_n 200 \
  --direction higher_is_better

cp "${OUT_DIR}/step09_mpnn_top200.csv" "${OUT_DIR}/step10_plip_msa_checked.csv"

python src/design_specific_enzymes/filtering_pipeline/scripts/filter_rosetta_gates.py \
  --input_csv "${OUT_DIR}/step10_plip_msa_checked.csv" \
  --output_csv "${OUT_DIR}/step11_rosetta_final.csv" \
  --rosetta_score_column rosetta_score \
  --relative_hydrophobic_sasa_column relative_hydrophobic_sasa \
  --net_charge_column net_charge \
  --average_sap_column average_sap \
  --hydrophobic_ratio_column hydrophobic_sasa_to_ideal_ratio \
  --relative_hydrophobic_sasa_max 3.0 \
  --average_sap_max 0.5 \
  --hydrophobic_ratio_max 1.7 \
  --require_nonzero_net_charge

echo "Filtering pipeline completed under ${OUT_DIR}"
