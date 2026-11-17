#!/usr/bin/env bash
set -euo pipefail

# Run the pre-analysis metric stage before filtering.
# Configure paths through environment variables as needed.

RUN_PERPLEXITY="${RUN_PERPLEXITY:-1}"
RUN_CLEAN="${RUN_CLEAN:-1}"
RUN_ALPHAFOLD="${RUN_ALPHAFOLD:-1}"
RUN_MMSEQS="${RUN_MMSEQS:-1}"
RUN_FOLDSEEK="${RUN_FOLDSEEK:-1}"
RUN_ESM1V="${RUN_ESM1V:-1}"
RUN_PROTEINMPNN="${RUN_PROTEINMPNN:-1}"
RUN_PLIP="${RUN_PLIP:-1}"
RUN_ROSETTA="${RUN_ROSETTA:-1}"

if [ "${RUN_PERPLEXITY}" = "1" ]; then
  bash src/design_specific_enzymes/filtering_pipeline/scripts/run_metric_perplexity.sh
fi
if [ "${RUN_CLEAN}" = "1" ]; then
  bash src/design_specific_enzymes/filtering_pipeline/scripts/run_metric_clean.sh
fi
if [ "${RUN_ALPHAFOLD}" = "1" ]; then
  bash src/design_specific_enzymes/filtering_pipeline/scripts/run_metric_alphafold.sh
fi
if [ "${RUN_MMSEQS}" = "1" ]; then
  bash src/design_specific_enzymes/filtering_pipeline/scripts/run_metric_mmseqs.sh
fi
if [ "${RUN_FOLDSEEK}" = "1" ]; then
  bash src/design_specific_enzymes/filtering_pipeline/scripts/run_metric_foldseek.sh
fi
if [ "${RUN_ESM1V}" = "1" ]; then
  bash src/design_specific_enzymes/filtering_pipeline/scripts/run_metric_esm1v.sh
fi
if [ "${RUN_PROTEINMPNN}" = "1" ]; then
  bash src/design_specific_enzymes/filtering_pipeline/scripts/run_metric_proteinmpnn.sh
fi
if [ "${RUN_PLIP}" = "1" ]; then
  bash src/design_specific_enzymes/filtering_pipeline/scripts/run_metric_plip.sh
fi
if [ "${RUN_ROSETTA}" = "1" ]; then
  bash src/design_specific_enzymes/filtering_pipeline/scripts/run_metric_rosetta.sh
fi

echo "Metric run stage completed."
