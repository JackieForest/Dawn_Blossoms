#!/bin/bash
# Split qwen35_4b_science_rl_step50_fixed rollout by benchmark, then verify each
# benchmark in an independent CompassVerifier job.
#
# Usage:
#   bash script/launch_qwen35_4b_science_rl_step50_verify_by_benchmark.sh
#   bash script/launch_qwen35_4b_science_rl_step50_verify_by_benchmark.sh ScienceQA_TEST
set -euo pipefail

PROJECT_DIR="/mnt/petrelfs/linjuekai/Multimodel_Reasoning/eval"
OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/eval}"
SOURCE_ALIAS="${SOURCE_ALIAS:-qwen35_4b_science_rl_step50_fixed}"
SUBMIT_SLEEP="${SUBMIT_SLEEP:-1}"

cd "${PROJECT_DIR}"
mkdir -p verify_logs

BENCHMARKS=(
  MMMU_DEV_VAL
  AI2D_TEST
  ScienceQA_TEST
  SFE
)

ONLY_BENCHMARK="${1:-}"
SOURCE_ROLLOUT="${OUTPUT_ROOT}/outputs/${SOURCE_ALIAS}/rollout.jsonl"

if [ ! -s "${SOURCE_ROLLOUT}" ]; then
  echo "Missing source rollout: ${SOURCE_ROLLOUT}" >&2
  exit 1
fi

split_one() {
  local benchmark="$1"
  local alias="${SOURCE_ALIAS}_${benchmark}"
  local output_path="${OUTPUT_ROOT}/outputs/${alias}"
  local rollout="${output_path}/rollout.jsonl"

  mkdir -p "${output_path}"
  BENCHMARK="${benchmark}" SOURCE_ROLLOUT="${SOURCE_ROLLOUT}" ROLLOUT="${rollout}" python - <<'PY'
import json
import os

benchmark = os.environ["BENCHMARK"]
source = os.environ["SOURCE_ROLLOUT"]
target = os.environ["ROLLOUT"]

count = 0
with open(source, "r", encoding="utf-8") as fin, open(target, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        item = json.loads(line)
        dataset = item.get("dataset") or item.get("dataset_name") or item.get("data_source")
        if dataset == benchmark:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

print(f"{benchmark}: wrote {count} items -> {target}")
if count == 0:
    raise SystemExit(2)
PY
}

launch_verify() {
  local benchmark="$1"
  local alias="${SOURCE_ALIAS}_${benchmark}"
  local output_path="${OUTPUT_ROOT}/outputs/${alias}"
  local log_path="verify_logs/${alias}.launcher.log"

  echo "Launching verify ${alias} -> ${log_path}"
  VERIFY_PARTITION="${VERIFY_PARTITION:-sciverse_agent}" \
  VERIFY_QUOTA="${VERIFY_QUOTA:-auto}" \
  VERIFY_GPUS="${VERIFY_GPUS:-4}" \
  VERIFY_VLLM_MAX_MODEL_LEN="${VERIFY_VLLM_MAX_MODEL_LEN:-32768}" \
  VERIFY_VLLM_GPU_MEMORY_UTILIZATION="${VERIFY_VLLM_GPU_MEMORY_UTILIZATION:-0.5}" \
  VERIFY_VLLM_ENFORCE_EAGER="${VERIFY_VLLM_ENFORCE_EAGER:-1}" \
  EVAL_OUTPUT_ROOT="${OUTPUT_ROOT}" \
  nohup bash verify.sh "${output_path}" > "${log_path}" 2>&1 &

  sleep "${SUBMIT_SLEEP}"
}

for benchmark in "${BENCHMARKS[@]}"; do
  if [ -n "${ONLY_BENCHMARK}" ] && [ "${benchmark}" != "${ONLY_BENCHMARK}" ]; then
    continue
  fi

  split_one "${benchmark}"
  launch_verify "${benchmark}"
done

echo "Submitted requested per-benchmark verify jobs."
echo "Check queue:"
echo "  squeue -u linjuekai -o '%.18i %.45j %.8T %.12M %.9l %.6D %R'"
echo "Launcher logs:"
echo "  ${PROJECT_DIR}/verify_logs/${SOURCE_ALIAS}_*.launcher.log"
echo "Verify logs:"
echo "  ${PROJECT_DIR}/verify_logs/${SOURCE_ALIAS}_*.log"
