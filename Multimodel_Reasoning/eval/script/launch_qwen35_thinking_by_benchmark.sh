#!/bin/bash
# Launch Qwen3.5 thinking eval split by benchmark.
# Each benchmark uses one independent srun job, one GPU, and then runs verify via eval/run.sh.
# Usage:
#   bash script/launch_qwen35_thinking_by_benchmark.sh          # launch 4B + 9B, 18 jobs each
#   bash script/launch_qwen35_thinking_by_benchmark.sh 4b       # launch only 4B
#   bash script/launch_qwen35_thinking_by_benchmark.sh 9b       # launch only 9B
#   bash script/launch_qwen35_thinking_by_benchmark.sh all SFE  # launch only selected benchmark for both
set -euo pipefail

PROJECT_DIR="/mnt/petrelfs/linjuekai/Multimodel_Reasoning/eval"
MODEL_ROOT="/mnt/dhwfile/raise/user/linjuekai/models"
cd "${PROJECT_DIR}"

mkdir -p launcher_logs server_logs distill_logs verify_logs

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

TARGET="${1:-all}"
ONLY_BENCHMARK="${2:-}"

launch_one() {
  local model_size="$1"
  local model_path="$2"
  local benchmark="$3"
  local alias="Qwen3.5-${model_size}-thinking_${benchmark}"
  local log_path="launcher_logs/${alias}.log"

  echo "Launching ${alias} -> ${log_path}"
  CLUSTER="${CLUSTER:-sciverse_agent}" \
  EVAL_QUOTA="${EVAL_QUOTA:-reserved}" \
  DATASET="${benchmark}" \
  EVAL_MODEL_ALIAS="${alias}" \
  EVAL_ENABLE_THINKING=1 \
  EVAL_SYSTEM_PROMPT_MODE="${EVAL_SYSTEM_PROMPT_MODE:-thinking}" \
  EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-16384}" \
  EVAL_REQUEST_TIMEOUT="${EVAL_REQUEST_TIMEOUT:-7200}" \
  EVAL_MAX_CONCURRENT="${EVAL_MAX_CONCURRENT:-32}" \
  EVAL_IMAGE_TARGET_SIZES="${EVAL_IMAGE_TARGET_SIZES:-1024,768,512}" \
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}" \
  VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}" \
  nohup bash run.sh "${model_path}" 1 > "${log_path}" 2>&1 &
}

for benchmark in "${BENCHMARKS[@]}"; do
  if [ -n "${ONLY_BENCHMARK}" ] && [ "${benchmark}" != "${ONLY_BENCHMARK}" ]; then
    continue
  fi

  case "${TARGET}" in
    all)
      launch_one "4B" "${MODEL_ROOT}/Qwen3.5-4B" "${benchmark}"
      launch_one "9B" "${MODEL_ROOT}/Qwen3.5-9B" "${benchmark}"
      ;;
    4b|4B)
      launch_one "4B" "${MODEL_ROOT}/Qwen3.5-4B" "${benchmark}"
      ;;
    9b|9B)
      launch_one "9B" "${MODEL_ROOT}/Qwen3.5-9B" "${benchmark}"
      ;;
    *)
      echo "Unknown target: ${TARGET}. Use: all, 4b, or 9b." >&2
      exit 1
      ;;
  esac

done

echo "Submitted requested jobs. Check with:"
echo "  squeue -u linjuekai -o '%.18i %.45j %.8T %.12M %.9l %.6D %R'"
echo "Logs: ${PROJECT_DIR}/launcher_logs/Qwen3.5-*-thinking_*.log"
