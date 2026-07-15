#!/bin/bash
# Launch CompassVerifier for Qwen3.5 thinking eval outputs split by benchmark.
# Each benchmark starts one independent background verify.sh process.
# Usage:
#   bash script/launch_qwen35_thinking_verify_by_benchmark.sh          # verify 4B + 9B, 18 jobs each
#   bash script/launch_qwen35_thinking_verify_by_benchmark.sh 4b       # verify only 4B
#   bash script/launch_qwen35_thinking_verify_by_benchmark.sh 9b       # verify only 9B
#   bash script/launch_qwen35_thinking_verify_by_benchmark.sh all SFE  # verify only selected benchmark for both
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

launch_one() {
  local model_size="$1"
  local benchmark="$2"
  local alias="Qwen3.5-${model_size}-thinking_${benchmark}"
  local output_path="${OUTPUT_ROOT}/outputs/${alias}"
  local log_path="verify_logs/${alias}.launcher.log"

  if [ ! -f "${output_path}/rollout.jsonl" ]; then
    echo "Skip ${alias}: missing ${output_path}/rollout.jsonl" >&2
    return 0
  fi

  echo "Launching verify ${alias} -> ${log_path}"
  VERIFY_PARTITION="${VERIFY_PARTITION:-sciverse_agent}" \
  VERIFY_QUOTA="${VERIFY_QUOTA:-reserved}" \
  VERIFY_GPUS="${VERIFY_GPUS:-1}" \
  EVAL_OUTPUT_ROOT="${OUTPUT_ROOT}" \
  nohup bash verify.sh "${output_path}" > "${log_path}" 2>&1 &
}

for benchmark in "${BENCHMARKS[@]}"; do
  if [ -n "${ONLY_BENCHMARK}" ] && [ "${benchmark}" != "${ONLY_BENCHMARK}" ]; then
    continue
  fi

  case "${TARGET}" in
    all)
      launch_one "4B" "${benchmark}"
      launch_one "9B" "${benchmark}"
      ;;
    4b|4B)
      launch_one "4B" "${benchmark}"
      ;;
    9b|9B)
      launch_one "9B" "${benchmark}"
      ;;
    *)
      echo "Unknown target: ${TARGET}. Use: all, 4b, or 9b." >&2
      exit 1
      ;;
  esac
done

echo "Submitted requested verify jobs. Check with:"
echo "  squeue -u linjuekai -o '%.18i %.45j %.8T %.12M %.9l %.6D %R'"
echo "Launcher logs: ${PROJECT_DIR}/verify_logs/Qwen3.5-*-thinking_*.launcher.log"
echo "Verify logs:   ${PROJECT_DIR}/verify_logs/Qwen3.5-*-thinking_*.log"
