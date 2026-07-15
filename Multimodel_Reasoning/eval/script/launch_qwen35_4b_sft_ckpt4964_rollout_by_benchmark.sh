#!/bin/bash
# Roll out the Qwen3.5-4B full-distill SFT checkpoint-4964 by benchmark.
# One benchmark = one single-GPU vLLM job, with isolated aliases/logs/outputs.
set -euo pipefail

PROJECT_DIR=/mnt/petrelfs/linjuekai/Multimodel_Reasoning/eval
MODEL_PATH=/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/training/models/qwen35_4b_base_full_distill_sft_528k_16k/checkpoint-4964
BASE_ALIAS=qwen35_4b_base_full_distill_sft_528k_16k_ckpt4964
SUBMIT_SLEEP="${SUBMIT_SLEEP:-0.5}"

BENCHMARKS=(
  SFE
  MathVision
  MMMU_DEV_VAL
  MathVista_MINI
  VisuLogic
  CharXiv_reasoning_val
  CharXiv_descriptive_val
  LogicVista
  ChartQA_TEST
  OCRBench
  RealWorldQA
  MathVerse_MINI
  AI2D_TEST
  ScienceQA_TEST
  MMBench_DEV_EN_V11
  MMStar
  CV-Bench-2D
  CV-Bench-3D
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

echo "Submitted checkpoint-4964 per-benchmark rollout jobs."
echo "Base alias: ${BASE_ALIAS}"
echo "Model: ${MODEL_PATH}"
echo "Logs:"
echo "  ${PROJECT_DIR}/launcher_logs/${BASE_ALIAS}_*.log"
echo "  ${PROJECT_DIR}/server_logs/${BASE_ALIAS}_*.log"
echo "  ${PROJECT_DIR}/distill_logs/${BASE_ALIAS}_*.log"
echo "Outputs:"
echo "  /mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/eval/outputs/${BASE_ALIAS}_*/rollout.jsonl"
