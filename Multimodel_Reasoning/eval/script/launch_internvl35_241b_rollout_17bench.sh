#!/bin/bash
# Launch InternVL3.5-241B-A28B rollout on 17 benchmarks, excluding OCRBench.
# This submits one 8-GPU job and only generates rollout.jsonl; verify is manual.
#
# Usage:
#   bash script/launch_internvl35_241b_rollout_17bench.sh
#   bash script/launch_internvl35_241b_rollout_17bench.sh
set -euo pipefail

PROJECT_DIR="/mnt/petrelfs/linjuekai/Multimodel_Reasoning/eval"
MODEL_PATH="${MODEL_PATH:-/mnt/dhwfile/zhangjunyuan/models/InternVL3_5-241B-A28B}"
ALIAS="${EVAL_MODEL_ALIAS:-InternVL3_5-241B-A28B_17bench_no_ocr}"

cd "${PROJECT_DIR}"
mkdir -p launcher_logs server_logs distill_logs verify_logs

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Missing model path: ${MODEL_PATH}" >&2
  exit 1
fi

DATASET_NO_OCR="SFE,MathVision,MMMU_DEV_VAL,MathVista_MINI,VisuLogic,CharXiv_reasoning_val,CharXiv_descriptive_val,LogicVista,ChartQA_TEST,RealWorldQA,MathVerse_MINI,AI2D_TEST,ScienceQA_TEST,MMBench_DEV_EN_V11,MMStar,CV-Bench-2D,CV-Bench-3D"
LOG_PATH="launcher_logs/${ALIAS}.log"

echo "Launching ${ALIAS} -> ${LOG_PATH}"
CLUSTER="${CLUSTER:-sciverse_agent}" \
EVAL_QUOTA="${EVAL_QUOTA:-reserved}" \
DATASET="${DATASET:-${DATASET_NO_OCR}}" \
EVAL_MODEL_ALIAS="${ALIAS}" \
EVAL_SRUN_ASYNC=1 \
EVAL_ENABLE_THINKING=0 \
EVAL_SYSTEM_PROMPT_MODE="" \
EVAL_SYSTEM_PROMPT="" \
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-4096}" \
EVAL_REQUEST_TIMEOUT="${EVAL_REQUEST_TIMEOUT:-7200}" \
EVAL_MAX_CONCURRENT="${EVAL_MAX_CONCURRENT:-32}" \
EVAL_IMAGE_TARGET_SIZES="${EVAL_IMAGE_TARGET_SIZES:--1,1024,768,512}" \
VLLM_SIF="${VLLM_SIF:-/mnt/dhwfile/raise/user/wangxiaoyang/apptainer-image/vllm-cu128.sif}" \
PATCH_LIB="${PATCH_LIB:-/mnt/petrelfs/linjuekai/Multimodel_Reasoning/eval/empty_patch_lib}" \
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}" \
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-512}" \
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}" \
VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}" \
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}" \
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-1800}" \
nohup bash run.sh "${MODEL_PATH}" 8 > "${LOG_PATH}" 2>&1 &

echo "Submitted launcher process."
echo "Launcher log: ${PROJECT_DIR}/${LOG_PATH}"
echo "Server log:   ${PROJECT_DIR}/server_logs/${ALIAS}.log"
echo "Rollout log:  ${PROJECT_DIR}/distill_logs/${ALIAS}.log"
echo "Output:       /mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/eval/outputs/${ALIAS}/rollout.jsonl"
echo "Queue:"
echo "  squeue -u linjuekai -o '%.18i %.55j %.8T %.12M %.9l %.6D %R'"
