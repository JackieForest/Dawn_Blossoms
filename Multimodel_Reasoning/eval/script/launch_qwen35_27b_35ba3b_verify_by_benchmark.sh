#!/bin/bash
# Launch CompassVerifier for Qwen3.5 27B/35B-A3B per-benchmark rollout outputs.
#
# Usage:
#   bash script/launch_qwen35_27b_35ba3b_verify_by_benchmark.sh
#   bash script/launch_qwen35_27b_35ba3b_verify_by_benchmark.sh 27b-thinking
#   bash script/launch_qwen35_27b_35ba3b_verify_by_benchmark.sh 35b-nothinking ScienceQA_TEST
set -euo pipefail

PROJECT_DIR="/mnt/petrelfs/linjuekai/Multimodel_Reasoning/eval"
OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/eval}"
cd "${PROJECT_DIR}"

mkdir -p verify_logs

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
SUBMIT_SLEEP="${SUBMIT_SLEEP:-1}"

prefixes_for_target() {
  case "$1" in
    all)
      printf '%s\n' \
        Qwen3.5-27B-thinking \
        Qwen3.5-35B-A3B-thinking \
        Qwen3.5-35B-A3B-nothinking
      ;;
    27b-thinking|27B-thinking)
      printf '%s\n' Qwen3.5-27B-thinking
      ;;
    35b-thinking|35B-thinking|35ba3b-thinking|35BA3B-thinking)
      printf '%s\n' Qwen3.5-35B-A3B-thinking
      ;;
    35b-nothinking|35B-nothinking|35ba3b-nothinking|35BA3B-nothinking)
      printf '%s\n' Qwen3.5-35B-A3B-nothinking
      ;;
    *)
      echo "Unknown target: $1" >&2
      echo "Use: all, 27b-thinking, 35b-thinking, or 35b-nothinking." >&2
      exit 1
      ;;
  esac
}

launch_verify() {
  local alias="$1"
  local output_path="${OUTPUT_ROOT}/outputs/${alias}"
  local rollout="${output_path}/rollout.jsonl"
  local log_path="verify_logs/${alias}.launcher.log"

  if [ ! -s "${rollout}" ]; then
    echo "Skip ${alias}: missing rollout ${rollout}" >&2
    return 0
  fi

  echo "Launching verify ${alias} -> ${log_path}"
  VERIFY_PARTITION="${VERIFY_PARTITION:-sciverse_agent}" \
  VERIFY_QUOTA="${VERIFY_QUOTA:-reserved}" \
  VERIFY_GPUS="${VERIFY_GPUS:-4}" \
  VERIFY_VLLM_MAX_MODEL_LEN="${VERIFY_VLLM_MAX_MODEL_LEN:-32768}" \
  VERIFY_VLLM_GPU_MEMORY_UTILIZATION="${VERIFY_VLLM_GPU_MEMORY_UTILIZATION:-0.5}" \
  VERIFY_VLLM_ENFORCE_EAGER="${VERIFY_VLLM_ENFORCE_EAGER:-1}" \
  EVAL_OUTPUT_ROOT="${OUTPUT_ROOT}" \
  nohup bash verify.sh "${output_path}" > "${log_path}" 2>&1 &

  sleep "${SUBMIT_SLEEP}"
}

while IFS= read -r prefix; do
  for benchmark in "${BENCHMARKS[@]}"; do
    if [ -n "${ONLY_BENCHMARK}" ] && [ "${benchmark}" != "${ONLY_BENCHMARK}" ]; then
      continue
    fi
    launch_verify "${prefix}_${benchmark}"
  done
done < <(prefixes_for_target "${TARGET}")

echo "Submitted requested verify jobs."
echo "Check queue:"
echo "  squeue -u linjuekai -o '%.18i %.45j %.8T %.12M %.9l %.6D %R'"
