#!/usr/bin/env bash
#SBATCH --job-name=mmr-gspo-qwenjudge-train
#SBATCH --partition=sciverse_agent
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --quotatype=reserved
#SBATCH --output=/mnt/petrelfs/linjuekai/Multimodel_Reasoning/rl/logs/gspo_qwenjudge_train_%x_%j.log

set -xeuo pipefail

DOMAIN=${1:-${DOMAIN:-science}}
RL_DIR=${RL_DIR:-/mnt/petrelfs/linjuekai/Multimodel_Reasoning/rl}
URL_FILE=${URL_FILE:-${RL_DIR}/qwen_judge_urls/${DOMAIN}.txt}

cd "${RL_DIR}"
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES
export NNODES=1
export N_GPUS_PER_NODE=8
export TRAIN_GPUS=0,1,2,3,4,5,6,7
export WANDB_MODE=offline

START_COMPASS=0 URL_FILE="${URL_FILE}" bash "${RL_DIR}/run_gspo_qwen_judge_domain.sh" "${DOMAIN}"
