#!/usr/bin/env bash
#SBATCH --job-name=mmr-gspo-qwenjudge
#SBATCH --partition=sciverse_agent
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:12
#SBATCH --cpus-per-task=96
#SBATCH --quotatype=reserved
#SBATCH --output=/mnt/petrelfs/linjuekai/Multimodel_Reasoning/rl/logs/gspo_qwenjudge_%x_%j.log

set -xeuo pipefail

DOMAIN=${1:-${DOMAIN:-science}}
cd /mnt/petrelfs/linjuekai/Multimodel_Reasoning/rl
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11
export TRAIN_GPUS=0,1,2,3,4,5,6,7
export VERIFY_GPUS=8,9,10,11
export N_GPUS_PER_NODE=8

bash /mnt/petrelfs/linjuekai/Multimodel_Reasoning/rl/run_gspo_qwen_judge_domain.sh "${DOMAIN}"
