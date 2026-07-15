#!/bin/bash
# Launch eval for Qwen3.5-2B-Base and Qwen3.5-9B-Base.
# Each model uses one GPU and runs the full benchmark suite with the
# same defaults that were used for Qwen3.5-4B-Base.
set -euo pipefail

PROJECT_DIR=/mnt/petrelfs/linjuekai/Multimodel_Reasoning/eval
cd "${PROJECT_DIR}"

mkdir -p launcher_logs

MODELS=(
  /mnt/dhwfile/raise/user/linjuekai/models/Qwen3.5-2B-Base
  /mnt/dhwfile/raise/user/linjuekai/models/Qwen3.5-9B-Base
)

for model_path in "${MODELS[@]}"; do
  model_name="$(basename "${model_path}")"
  log_path="launcher_logs/${model_name}.log"

  if [ ! -d "${model_path}" ]; then
    echo "Missing model path: ${model_path}" >&2
    exit 1
  fi

  echo "Launching ${model_name} -> ${log_path}"
  CLUSTER="${CLUSTER:-sciverse_agent}" \
  EVAL_SRUN_ASYNC=1 \
  EVAL_QUOTA="${EVAL_QUOTA:-reserved}" \
  EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-16384}" \
  EVAL_REQUEST_TIMEOUT="${EVAL_REQUEST_TIMEOUT:-1800}" \
  EVAL_MAX_CONCURRENT="${EVAL_MAX_CONCURRENT:-250}" \
  EVAL_REPETITION_PENALTY="${EVAL_REPETITION_PENALTY:-1.05}" \
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}" \
  VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-512}" \
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}" \
  EVAL_MODEL_ALIAS="${model_name}" \
  nohup bash run.sh "${model_path}" 1 > "${log_path}" 2>&1 &
done

echo "Submitted base eval jobs. Check with:"
echo "  squeue -u linjuekai"
echo "Launcher logs: ${PROJECT_DIR}/launcher_logs/Qwen3.5-{2B,9B}-Base.log"
echo "Rollout outputs: /mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/eval/outputs/Qwen3.5-{2B,9B}-Base"
