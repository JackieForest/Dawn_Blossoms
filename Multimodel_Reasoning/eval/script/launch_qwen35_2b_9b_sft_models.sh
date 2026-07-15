#!/bin/bash
# Launch eval for Qwen3.5-2B and Qwen3.5-9B SFT models using the SFT prompt.
# Each model uses one GPU and runs the full 18-benchmark suite.
# These SFT checkpoints use a Qwen chat template where enable_thinking=true
# pre-fills "<think>\n"; enable_thinking=false pre-fills an empty think block.
# The vLLM defaults below are aligned with the earlier
# qwen35_4b_base_9b_mix_16k run for apples-to-apples comparison.
set -euo pipefail

PROJECT_DIR=/mnt/petrelfs/linjuekai/Multimodel_Reasoning/eval
MODEL_ROOT=/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/training/models
cd "${PROJECT_DIR}"

mkdir -p launcher_logs

EVAL_ENABLE_THINKING="${EVAL_ENABLE_THINKING:-1}"
EVAL_SYSTEM_PROMPT_MODE="${EVAL_SYSTEM_PROMPT_MODE:-sft}"
EVAL_RUN_SUFFIX="${EVAL_RUN_SUFFIX:-}"

MODELS=(
  qwen35_2b_base_122b_distill_sft_49k_16k
  qwen35_2b_base_27b_distill_sft_124k_16k
  qwen35_2b_base_9b_distill_sft_355k_16k
  qwen35_2b_base_full_distill_sft_528k_16k
  qwen35_9b_base_122b_distill_sft_49k_16k
  qwen35_9b_base_27b_distill_sft_124k_16k
  qwen35_9b_base_9b_distill_sft_355k_16k
  qwen35_9b_base_full_distill_sft_528k_16k
)

for model in "${MODELS[@]}"; do
  model_path="${MODEL_ROOT}/${model}"
  log_path="launcher_logs/${model}.log"

  if [ ! -d "${model_path}" ]; then
    echo "Missing model path: ${model_path}" >&2
    exit 1
  fi

  echo "Launching ${model} -> ${log_path}"
  CLUSTER="${CLUSTER:-sciverse_agent}" \
  EVAL_SRUN_ASYNC=1 \
  EVAL_QUOTA="${EVAL_QUOTA:-reserved}" \
  EVAL_SYSTEM_PROMPT_MODE="${EVAL_SYSTEM_PROMPT_MODE}" \
  EVAL_ENABLE_THINKING="${EVAL_ENABLE_THINKING}" \
  EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-16384}" \
  EVAL_REQUEST_TIMEOUT="${EVAL_REQUEST_TIMEOUT:-7200}" \
  EVAL_MAX_CONCURRENT="${EVAL_MAX_CONCURRENT:-32}" \
  EVAL_IMAGE_TARGET_SIZES="${EVAL_IMAGE_TARGET_SIZES:-1024,768,512}" \
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-50000}" \
  VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}" \
  EVAL_MODEL_ALIAS="${model}" \
  nohup bash run.sh "${model_path}" 1 > "${log_path}" 2>&1 &
done

echo "Submitted eight SFT eval jobs. Check with:"
echo "  squeue -u linjuekai"
echo "Launcher logs: ${PROJECT_DIR}/launcher_logs/qwen35_{2b,9b}_base_*_sft_*_16k.log"
echo "EVAL_ENABLE_THINKING=${EVAL_ENABLE_THINKING}"
echo "EVAL_SYSTEM_PROMPT_MODE=${EVAL_SYSTEM_PROMPT_MODE}"
echo "Rollout outputs: /mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/eval/outputs/qwen35_{2b,9b}_base_*_16k"
