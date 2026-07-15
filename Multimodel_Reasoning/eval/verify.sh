#!/bin/bash
# Verify rollout.jsonl with CompassVerifier.
# Usage: bash verify.sh <OUTPUT_PATH>
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJECT_DIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/eval}"
export EVAL_CACHE_ROOT="${EVAL_CACHE_ROOT:-${EVAL_OUTPUT_ROOT}/cache}"
PARTITION="${VERIFY_PARTITION:-sciverse_agent}"
QUOTA_TYPE="${VERIFY_QUOTA:-auto}"
GPUS_PER_NODE="${VERIFY_GPUS:-4}"
VLLM_SIF="${VERIFY_SIF:-/mnt/dhwfile/raise/user/caimengzhang/env/vmleval.sif}"
VERIFIER_MODEL_PATH="${VERIFIER_MODEL_PATH:-/mnt/dhwfile/raise/user/caimengzhang/huggingface/hub/models--opencompass--CompassVerifier-7B/snapshots/676c83e3c62c199e0d6ad29cd31b6064c8d500a0}"
VERIFY_SCRIPT="${VERIFY_SCRIPT:-/mnt/petrelfs/linjuekai/Multimodel_Reasoning/eval_thinking/verify.py}"
VERIFY_EXCLUDE_DEFAULT=""
VERIFY_EXCLUDE_NODES="${VERIFY_EXCLUDE_NODES:-${VERIFY_EXCLUDE_DEFAULT}}"
if [ -n "${VERIFY_EXCLUDE_EXTRA:-}" ]; then
    if [ -n "${VERIFY_EXCLUDE_NODES}" ]; then
        VERIFY_EXCLUDE_NODES="${VERIFY_EXCLUDE_NODES},${VERIFY_EXCLUDE_EXTRA}"
    else
        VERIFY_EXCLUDE_NODES="${VERIFY_EXCLUDE_EXTRA}"
    fi
fi
SRUN_EXCLUDE_ARGS=""
if [ -n "${VERIFY_EXCLUDE_NODES}" ]; then
    SRUN_EXCLUDE_ARGS="-x ${VERIFY_EXCLUDE_NODES}"
fi

MODEL_PATH="${1:?Usage: bash verify.sh <OUTPUT_PATH>}"
MODEL_BASENAME="$(basename "${MODEL_PATH}")"
export MODEL_PATH MODEL_BASENAME VLLM_SIF VERIFIER_MODEL_PATH GPUS_PER_NODE VERIFY_SCRIPT

mkdir -p verify_logs
mkdir -p "${EVAL_OUTPUT_ROOT}/outputs" "${EVAL_CACHE_ROOT}/hf" "${EVAL_CACHE_ROOT}/xdg/verify_${MODEL_BASENAME}" "${EVAL_CACHE_ROOT}/triton/verify_${MODEL_BASENAME}"
echo "VERIFY_PARTITION=${PARTITION}"
echo "VERIFY_QUOTA=${QUOTA_TYPE}"
echo "VERIFY_GPUS=${GPUS_PER_NODE}"
echo "VERIFY_VLLM_MAX_MODEL_LEN=${VERIFY_VLLM_MAX_MODEL_LEN:-32768}"
echo "VERIFY_VLLM_GPU_MEMORY_UTILIZATION=${VERIFY_VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
echo "VERIFY_VLLM_ENFORCE_EAGER=${VERIFY_VLLM_ENFORCE_EAGER:-1}"
echo "VERIFY_EXCLUDE_NODES=${VERIFY_EXCLUDE_NODES:-<none>}"
SRUN_CMD=(srun -p "${PARTITION}")
if [ "${VERIFY_SRUN_ASYNC:-0}" = "1" ] || [ "${VERIFY_SRUN_ASYNC:-}" = "true" ]; then
    SRUN_CMD+=(--async)
fi

"${SRUN_CMD[@]}" --quotatype="${QUOTA_TYPE}" --gres="gpu:${GPUS_PER_NODE}" --job-name="verify_${MODEL_BASENAME}" \
    ${SRUN_EXCLUDE_ARGS} \
    bash -c '
    set -euo pipefail
    echo "Verify node: $(hostname)"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
    apptainer exec --nv --cleanenv --bind /share:/share,/mnt:/mnt \
    --env HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
    --env EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}" \
    --env HF_HOME="${EVAL_CACHE_ROOT}/hf" \
    --env HF_DATASETS_CACHE="${EVAL_CACHE_ROOT}/hf/datasets" \
    --env XDG_CACHE_HOME="${EVAL_CACHE_ROOT}/xdg/verify_${MODEL_BASENAME}" \
    --env TRITON_CACHE_DIR="${EVAL_CACHE_ROOT}/triton/verify_${MODEL_BASENAME}" \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
    --env VERIFY_VLLM_MAX_MODEL_LEN="${VERIFY_VLLM_MAX_MODEL_LEN:-32768}" \
    --env VERIFY_VLLM_GPU_MEMORY_UTILIZATION="${VERIFY_VLLM_GPU_MEMORY_UTILIZATION:-0.5}" \
    --env VERIFY_VLLM_ENFORCE_EAGER="${VERIFY_VLLM_ENFORCE_EAGER:-1}" \
    --env ALL_PROXY= --env all_proxy= --env HTTP_PROXY= --env HTTPS_PROXY= \
    --env http_proxy= --env https_proxy= \
    "${VLLM_SIF}" \
    python "${VERIFY_SCRIPT}" --model_name "${MODEL_PATH}" --verifier_model_path "${VERIFIER_MODEL_PATH}" --tensor_parallel_size "${GPUS_PER_NODE}" \
    ' \
    2>&1 | tee "verify_logs/${MODEL_BASENAME}.log"

if [ "${VERIFY_SRUN_ASYNC:-0}" = "1" ] || [ "${VERIFY_SRUN_ASYNC:-}" = "true" ]; then
    echo "Submitted async verify job for ${MODEL_BASENAME}. Merge manually after verification finishes:"
    echo "  EVAL_OUTPUT_ROOT=\"${EVAL_OUTPUT_ROOT}\" python merge.py --model_name \"${MODEL_BASENAME}\""
    exit 0
fi

EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}" python merge.py --model_name "${MODEL_BASENAME}"
