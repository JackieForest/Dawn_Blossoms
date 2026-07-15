#!/usr/bin/env bash
set -euo pipefail

DOMAINS=(
  science
  chart_table_doc
  math
  logic_game_puzzle
  spatial_general
)

for domain in "${DOMAINS[@]}"; do
  echo "submit ${domain}"
  sbatch /mnt/petrelfs/linjuekai/Multimodel_Reasoning/rl/submit_gspo_compass_domain.sh "${domain}"
done
