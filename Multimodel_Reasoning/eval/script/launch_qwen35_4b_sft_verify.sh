#!/bin/bash
# Launch verify for the four Qwen3.5-4B SFT models using the same
# CompassVerifier TP=4 setup as the earlier qwen35_4b_base_9b_mix_16k run.
set -euo pipefail

PROJECT_DIR=/mnt/petrelfs/linjuekai/Multimodel_Reasoning/eval
OUTPUT_ROOT=/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/eval/outputs
cd "${PROJECT_DIR}"

mkdir -p verify_logs

MODELS=(
  qwen35_4b_base_122b_distill_sft_49k_16k
  qwen35_4b_base_27b_distill_sft_124k_16k
  qwen35_4b_base_9b_distill_sft_355k_16k
  qwen35_4b_base_full_distill_sft_528k_16k
)

for model in "${MODELS[@]}"; do
  output_path="${OUTPUT_ROOT}/${model}"
  log_path="verify_logs/${model}.launcher.log"

  if [ ! -d "${output_path}" ]; then
    echo "Missing output path: ${output_path}" >&2
    exit 1
  fi

  echo "Launching verify for ${model} -> ${log_path}"
  VERIFY_PARTITION="${VERIFY_PARTITION:-sciverse_agent}" \
  VERIFY_QUOTA="${VERIFY_QUOTA:-reserved}" \
  VERIFY_GPUS="${VERIFY_GPUS:-4}" \
  VERIFY_VLLM_MAX_MODEL_LEN="${VERIFY_VLLM_MAX_MODEL_LEN:-32768}" \
  VERIFY_VLLM_GPU_MEMORY_UTILIZATION="${VERIFY_VLLM_GPU_MEMORY_UTILIZATION:-0.5}" \
  VERIFY_VLLM_ENFORCE_EAGER="${VERIFY_VLLM_ENFORCE_EAGER:-1}" \
  nohup bash verify.sh "${output_path}" > "${log_path}" 2>&1 &
done

echo "Submitted four SFT verify jobs with mix_16k-style TP=4 defaults."
echo "Check with:"
echo "  squeue -u linjuekai | rg 'verify_qwen35_4b_base_'"
echo "Launcher logs: ${PROJECT_DIR}/verify_logs/qwen35_4b_base_*_sft_*_16k.launcher.log"
