#!/bin/bash
# Split the eight Qwen3.5 2B/9B SFT rollouts by benchmark, then verify each
# benchmark independently with CompassVerifier.
#
# This script only reads the original rollout.jsonl files. Split outputs are
# written to:
#   ${EVAL_OUTPUT_ROOT}/outputs/${model}_${benchmark}/rollout.jsonl
#
# Usage:
#   bash script/launch_qwen35_2b_9b_sft_verify_by_benchmark.sh
#   bash script/launch_qwen35_2b_9b_sft_verify_by_benchmark.sh qwen35_2b_base_9b_distill_sft_355k_16k
#   bash script/launch_qwen35_2b_9b_sft_verify_by_benchmark.sh all ScienceQA_TEST
#   VERIFY_GPUS=1 bash script/launch_qwen35_2b_9b_sft_verify_by_benchmark.sh
set -euo pipefail

PROJECT_DIR="/mnt/petrelfs/linjuekai/Multimodel_Reasoning/eval"
OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/eval}"
SUBMIT_SLEEP="${SUBMIT_SLEEP:-0.5}"
cd "${PROJECT_DIR}"

mkdir -p verify_logs

MODELS=(
  Qwen3.5-2B-Base
  qwen35_2b_base_9b_distill_sft_355k_16k
  qwen35_2b_base_27b_distill_sft_124k_16k
  qwen35_2b_base_122b_distill_sft_49k_16k
  qwen35_2b_base_full_distill_sft_528k_16k
  qwen35_9b_base_9b_distill_sft_355k_16k
  qwen35_9b_base_27b_distill_sft_124k_16k
  qwen35_9b_base_122b_distill_sft_49k_16k
  qwen35_9b_base_full_distill_sft_528k_16k
)

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

TARGET_MODEL="${1:-all}"
ONLY_BENCHMARK="${2:-}"

split_rollout() {
  local model="$1"
  local benchmark="$2"
  local src="${OUTPUT_ROOT}/outputs/${model}/rollout.jsonl"
  local dst_dir="${OUTPUT_ROOT}/outputs/${model}_${benchmark}"
  local dst="${dst_dir}/rollout.jsonl"

  if [ ! -f "${src}" ]; then
    echo "Missing source rollout: ${src}" >&2
    return 1
  fi

  if [ -s "${dst}" ] && [ "${FORCE_SPLIT:-0}" != "1" ]; then
    echo "Reuse split rollout: ${dst}"
    return 0
  fi

  mkdir -p "${dst_dir}"
  SRC="${src}" DST="${dst}" BENCHMARK="${benchmark}" python3 - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ["SRC"])
dst = Path(os.environ["DST"])
benchmark = os.environ["BENCHMARK"]
tmp = dst.with_suffix(dst.suffix + ".tmp")

count = 0
with src.open("r", encoding="utf-8") as f, tmp.open("w", encoding="utf-8") as w:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("dataset") == benchmark:
            w.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

if count == 0:
    tmp.unlink(missing_ok=True)
    raise SystemExit(f"No rows for benchmark {benchmark} in {src}")

tmp.replace(dst)
print(f"Wrote {count} rows -> {dst}")
PY
}

launch_verify() {
  local model="$1"
  local benchmark="$2"
  local alias="${model}_${benchmark}"
  local output_path="${OUTPUT_ROOT}/outputs/${alias}"
  local log_path="verify_logs/${alias}.launcher.log"

  split_rollout "${model}" "${benchmark}"

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

for model in "${MODELS[@]}"; do
  if [ "${TARGET_MODEL}" != "all" ] && [ "${TARGET_MODEL}" != "${model}" ]; then
    continue
  fi

  for benchmark in "${BENCHMARKS[@]}"; do
    if [ -n "${ONLY_BENCHMARK}" ] && [ "${ONLY_BENCHMARK}" != "${benchmark}" ]; then
      continue
    fi

    launch_verify "${model}" "${benchmark}"
  done
done

echo "Submitted requested SFT per-benchmark verify jobs."
echo "Defaults: VERIFY_GPUS=${VERIFY_GPUS:-4}, SUBMIT_SLEEP=${SUBMIT_SLEEP}"
echo "Check queue:"
echo "  squeue -u linjuekai -o '%.18i %.45j %.8T %.12M %.9l %.6D %R'"
echo "Launcher logs:"
echo "  ${PROJECT_DIR}/verify_logs/qwen35_{2b,9b}_base_*_sft_*_16k_*.launcher.log"
