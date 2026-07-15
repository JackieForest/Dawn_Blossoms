#!/usr/bin/env bash
#SBATCH --job-name=mmr-qwenjudge-serve
#SBATCH --partition=sciverse_agent
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --quotatype=reserved
#SBATCH --output=/mnt/petrelfs/linjuekai/Multimodel_Reasoning/rl/logs/serve_qwenjudge_%x_%j.log

set -xeuo pipefail

DOMAIN=${1:-${DOMAIN:-science}}
RL_DIR=${RL_DIR:-/mnt/petrelfs/linjuekai/Multimodel_Reasoning/rl}
URL_FILE=${URL_FILE:-${RL_DIR}/qwen_judge_urls/${DOMAIN}.txt}
PORT=${PORT:-$((18765 + (${SLURM_JOB_ID:-0} % 1000)))}

cd "${RL_DIR}"
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES

mkdir -p "${RL_DIR}/logs" "${RL_DIR}/qwen_judge_urls"
TP=${QWEN_JUDGE_TP:-4} DP=${QWEN_JUDGE_DP:-1} URL_FILE="${URL_FILE}" PORT="${PORT}" bash "${RL_DIR}/serve_qwen_judge.sh"
