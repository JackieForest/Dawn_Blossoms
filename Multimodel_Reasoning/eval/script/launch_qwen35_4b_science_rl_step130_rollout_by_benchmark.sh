#!/bin/bash
# Roll out science RL step130 on the four science benchmarks.
set -euo pipefail

PROJECT_DIR=/mnt/petrelfs/linjuekai/Multimodel_Reasoning/eval
MODEL_PATH=/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/training/rl/qwen35_4b_full_sft_science_gspo_compass/global_step_130/actor_hf_merged_science_style_rl_lmhead
BASE_ALIAS=qwen35_4b_science_rl_step130
SUBMIT_SLEEP="${SUBMIT_SLEEP:-0.5}"

BENCHMARKS=(
  MMMU_DEV_VAL
  AI2D_TEST
  ScienceQA_TEST
  SFE
)

ONLY_BENCHMARK="${1:-}"

cd "${PROJECT_DIR}"
mkdir -p launcher_logs server_logs distill_logs

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Missing model path: ${MODEL_PATH}" >&2
  exit 1
fi

for benchmark in "${BENCHMARKS[@]}"; do
  if [ -n "${ONLY_BENCHMARK}" ] && [ "${ONLY_BENCHMARK}" != "${benchmark}" ]; then
    continue
  fi

  alias="${BASE_ALIAS}_${benchmark}"
  log_path="launcher_logs/${alias}.log"

  echo "Launching rollout ${alias} -> ${log_path}"
  CLUSTER="${CLUSTER:-sciverse_agent}" \
  EVAL_SRUN_ASYNC=1 \
  EVAL_QUOTA="${EVAL_QUOTA:-reserved}" \
  DATASET="${benchmark}" \
  EVAL_MODEL_ALIAS="${alias}" \
  EVAL_SYSTEM_PROMPT_MODE="${EVAL_SYSTEM_PROMPT_MODE:-sft}" \
  EVAL_ENABLE_THINKING="${EVAL_ENABLE_THINKING:-1}" \
  EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-16384}" \
  EVAL_REQUEST_TIMEOUT="${EVAL_REQUEST_TIMEOUT:-7200}" \
  EVAL_MAX_CONCURRENT="${EVAL_MAX_CONCURRENT:-64}" \
  EVAL_IMAGE_TARGET_SIZES="${EVAL_IMAGE_TARGET_SIZES:-1024,768,512}" \
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-50000}" \
  VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}" \
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}" \
  VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}" \
  nohup bash run.sh "${MODEL_PATH}" "${NUM_GPUS:-1}" > "${log_path}" 2>&1 &

  sleep "${SUBMIT_SLEEP}"
done

echo "Submitted science step130 per-benchmark rollout jobs."
echo "Base alias: ${BASE_ALIAS}"
echo "Model: ${MODEL_PATH}"
echo "Outputs: /mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/eval/outputs/${BASE_ALIAS}_*/rollout.jsonl"
