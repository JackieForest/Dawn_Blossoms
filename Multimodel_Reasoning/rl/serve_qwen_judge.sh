#!/usr/bin/env bash
set -xeuo pipefail

RL_DIR=${RL_DIR:-/mnt/petrelfs/linjuekai/Multimodel_Reasoning/rl}
SIF=${SIF:-/mnt/dhwfile/raise/user/linhonglin/apptainer/vllm-v0.17.1.sif}
OVERLAY=${OVERLAY:-/mnt/dhwfile/raise/user/linhonglin/apptainer/verl_extra}
HF_HOME=${HF_HOME:-/mnt/dhwfile/raise/user/linhonglin/hf/huggingface}

QWEN_JUDGE_MODEL_PATH=${QWEN_JUDGE_MODEL_PATH:-/mnt/dhwfile/raise/user/linjuekai/models/Qwen3.5-27B}
SERVED_NAME=${SERVED_NAME:-qwen3.5-27b-judge}
PORT=${PORT:-18765}
TP=${TP:-4}
DP=${DP:-1}
URL_FILE=${URL_FILE:-${RL_DIR}/qwen_judge_urls/judge_url.txt}

mkdir -p "$(dirname "${URL_FILE}")"
IP=$(hostname -I | awk '{print $1}')
URL="http://${IP}:${PORT}/v1"
echo "${URL}" > "${URL_FILE}"
echo "[serve_qwen_judge] URL=${URL}"

apptainer exec --nv --cleanenv \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    --env PYTHONPATH="${OVERLAY}" \
    --env HF_HOME="${HF_HOME}" \
    --env HF_HUB_OFFLINE=1 \
    --env VLLM_LOGGING_LEVEL=WARNING \
    --bind /mnt:/mnt \
    "${SIF}" \
    python -m vllm.entrypoints.openai.api_server \
        --model "${QWEN_JUDGE_MODEL_PATH}" \
        --served-model-name "${SERVED_NAME}" \
        --tensor-parallel-size "${TP}" \
        --data-parallel-size "${DP}" \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --enforce-eager \
        --gpu-memory-utilization "${QWEN_JUDGE_GPU_MEMORY_UTILIZATION:-0.8}" \
        --max-model-len "${QWEN_JUDGE_MAX_MODEL_LEN:-32768}" \
        --generation-config vllm \
        --no-enable-log-requests
