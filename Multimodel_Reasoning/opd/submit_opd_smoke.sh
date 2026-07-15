#!/usr/bin/env bash
#SBATCH --job-name=mmr-opd-smoke
#SBATCH --partition=sciverse_agent
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --quotatype=reserved
#SBATCH --output=/mnt/petrelfs/linjuekai/Multimodel_Reasoning/opd/logs/opd_smoke_%x_%j.log

set -xeuo pipefail

OPD_DIR=${OPD_DIR:-/mnt/petrelfs/linjuekai/Multimodel_Reasoning/opd}

cd "${OPD_DIR}"
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES

export NNODES=1
# verl's trainer.n_gpus_per_node is the student/main resource pool size.
# Distillation teachers use a separate pool below, so 2 + 2 = 4 Slurm GPUs.
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TRAIN_GPUS=${TRAIN_GPUS:-${CUDA_VISIBLE_DEVICES}}
export WANDB_MODE=${WANDB_MODE:-offline}

# Keep this submit script as a smoke test by default. Override any of these
# at sbatch time, e.g. `sbatch --export=ALL,TRAIN_STEPS=10 submit_opd_smoke.sh`.
export USE_SMOKE_DATA=${USE_SMOKE_DATA:-1}
export SMOKE_ROWS_PER_DOMAIN=${SMOKE_ROWS_PER_DOMAIN:-16}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
export TRAIN_STEPS=${TRAIN_STEPS:-3}
export SAVE_FREQ=${SAVE_FREQ:--1}
export TEACHER_NGPUS=${TEACHER_NGPUS:-2}
export TEACHER_TP=${TEACHER_TP:-2}

bash "${OPD_DIR}/run_opd_smoke.sh"
