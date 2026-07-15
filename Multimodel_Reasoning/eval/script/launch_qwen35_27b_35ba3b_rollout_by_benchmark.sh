#!/bin/bash
# Launch Qwen3.5 27B/35B-A3B rollout split by benchmark.
# This script only runs rollout generation. It does not run CompassVerifier.
#
# Usage:
#   bash script/launch_qwen35_27b_35ba3b_rollout_by_benchmark.sh
#   bash script/launch_qwen35_27b_35ba3b_rollout_by_benchmark.sh 27b-thinking
#   bash script/launch_qwen35_27b_35ba3b_rollout_by_benchmark.sh 35b-thinking ScienceQA_TEST
#   EVAL_GPUS=2 bash script/launch_qwen35_27b_35ba3b_rollout_by_benchmark.sh all
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
EVAL_GPUS="${EVAL_GPUS:-2}"
SUBMIT_SLEEP="${SUBMIT_SLEEP:-1}"

launch_one() {
  local alias_prefix="$1"
  local model_path="$2"
  local enable_thinking="$3"
  local system_prompt_mode="$4"
  local benchmark="$5"
  local alias="${alias_prefix}_${benchmark}"
  local log_path="launcher_logs/${alias}.log"

  if [ ! -d "${model_path}" ]; then
    echo "Missing model path: ${model_path}" >&2
    return 1
  fi

  echo "Launching ${alias} (${EVAL_GPUS} GPUs) -> ${log_path}"
  CLUSTER="${CLUSTER:-sciverse_agent}" \
  EVAL_QUOTA="${EVAL_QUOTA:-reserved}" \
  DATASET="${benchmark}" \
  EVAL_MODEL_ALIAS="${alias}" \
  EVAL_ENABLE_THINKING="${enable_thinking}" \
  EVAL_SYSTEM_PROMPT_MODE="${system_prompt_mode}" \
  EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-16384}" \
  EVAL_REQUEST_TIMEOUT="${EVAL_REQUEST_TIMEOUT:-7200}" \
  EVAL_MAX_CONCURRENT="${EVAL_MAX_CONCURRENT:-32}" \
  EVAL_IMAGE_TARGET_SIZES="${EVAL_IMAGE_TARGET_SIZES:-1024,768,512}" \
  EVAL_SRUN_ASYNC=1 \
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}" \
  VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}" \
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}" \
  VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}" \
  nohup bash run.sh "${model_path}" "${EVAL_GPUS}" > "${log_path}" 2>&1 &

  sleep "${SUBMIT_SLEEP}"
}

launch_target_for_benchmark() {
  local target="$1"
  local benchmark="$2"

  case "${target}" in
    all)
      launch_one "Qwen3.5-27B-thinking" "${MODEL_ROOT}/Qwen3.5-27B" 1 thinking "${benchmark}"
      launch_one "Qwen3.5-35B-A3B-thinking" "${MODEL_ROOT}/Qwen3.5-35B-A3B" 1 thinking "${benchmark}"
      launch_one "Qwen3.5-35B-A3B-nothinking" "${MODEL_ROOT}/Qwen3.5-35B-A3B" 0 "" "${benchmark}"
      ;;
    27b-thinking|27B-thinking)
      launch_one "Qwen3.5-27B-thinking" "${MODEL_ROOT}/Qwen3.5-27B" 1 thinking "${benchmark}"
      ;;
    35b-thinking|35B-thinking|35ba3b-thinking|35BA3B-thinking)
      launch_one "Qwen3.5-35B-A3B-thinking" "${MODEL_ROOT}/Qwen3.5-35B-A3B" 1 thinking "${benchmark}"
      ;;
    35b-nothinking|35B-nothinking|35ba3b-nothinking|35BA3B-nothinking)
      launch_one "Qwen3.5-35B-A3B-nothinking" "${MODEL_ROOT}/Qwen3.5-35B-A3B" 0 "" "${benchmark}"
      ;;
    *)
      echo "Unknown target: ${target}" >&2
      echo "Use: all, 27b-thinking, 35b-thinking, or 35b-nothinking." >&2
      exit 1
      ;;
  esac
}

for benchmark in "${BENCHMARKS[@]}"; do
  if [ -n "${ONLY_BENCHMARK}" ] && [ "${benchmark}" != "${ONLY_BENCHMARK}" ]; then
    continue
  fi
  launch_target_for_benchmark "${TARGET}" "${benchmark}"
done

echo "Submitted requested rollout jobs."
echo "Defaults: EVAL_GPUS=${EVAL_GPUS}, EVAL_MAX_CONCURRENT=${EVAL_MAX_CONCURRENT:-32}, VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-64}"
echo "Check queue:"
echo "  squeue -u linjuekai -o '%.18i %.45j %.8T %.12M %.9l %.6D %R'"
echo "Launcher logs:"
echo "  ${PROJECT_DIR}/launcher_logs/Qwen3.5-{27B,35B-A3B}-*.log"
echo "Rollout outputs:"
echo "  /mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/eval/outputs/Qwen3.5-<model>-<mode>_<benchmark>/rollout.jsonl"
