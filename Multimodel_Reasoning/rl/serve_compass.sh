#!/usr/bin/env bash
set -xeuo pipefail

RL_DIR=${RL_DIR:-/mnt/petrelfs/linjuekai/Multimodel_Reasoning/rl}
SIF=${SIF:-/mnt/dhwfile/raise/user/linhonglin/apptainer/vllm-v0.17.1.sif}
OVERLAY=${OVERLAY:-/mnt/dhwfile/raise/user/linhonglin/apptainer/verl_extra}
HF_HOME=${HF_HOME:-/mnt/dhwfile/raise/user/linhonglin/hf/huggingface}

COMPASS_MODEL_PATH=${COMPASS_MODEL_PATH:-/mnt/dhwfile/raise/user/linhonglin/hf/huggingface/hub/models--opencompass--CompassVerifier-7B/snapshots/676c83e3c62c199e0d6ad29cd31b6064c8d500a0}
SERVED_NAME=${SERVED_NAME:-opencompass/CompassVerifier-7B}
PORT=${PORT:-18765}
TP=${TP:-4}
DP=${DP:-1}
URL_FILE=${URL_FILE:-${RL_DIR}/compass_url.txt}

mkdir -p "$(dirname "${URL_FILE}")"
IP=$(hostname -I | awk '{print $1}')
URL="http://${IP}:${PORT}/v1"
echo "${URL}" > "${URL_FILE}"
echo "[serve_compass] URL=${URL}"

apptainer exec --nv --cleanenv \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    --env PYTHONPATH="${OVERLAY}" \
    --env HF_HOME="${HF_HOME}" \
    --env HF_HUB_OFFLINE=1 \
    --env VLLM_LOGGING_LEVEL=WARNING \
    --bind /mnt:/mnt \
    "${SIF}" \
    python -m vllm.entrypoints.openai.api_server \
        --model "${COMPASS_MODEL_PATH}" \
        --served-model-name "${SERVED_NAME}" \
        --tensor-parallel-size "${TP}" \
        --data-parallel-size "${DP}" \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --enforce-eager \
        --gpu-memory-utilization "${COMPASS_GPU_MEMORY_UTILIZATION:-0.8}" \
        --max-model-len "${COMPASS_MAX_MODEL_LEN:-32768}" \
        --no-enable-log-requests
