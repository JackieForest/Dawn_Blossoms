#!/bin/bash
# Run 18-benchmark VLM evaluation, then verify with CompassVerifier.
# Usage: bash run.sh <MODEL_PATH> [NUM_GPUS]
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJECT_DIR}"

export DATASET="${DATASET:-SFE,MathVision,MMMU_DEV_VAL,MathVista_MINI,VisuLogic,CharXiv_reasoning_val,CharXiv_descriptive_val,LogicVista,ChartQA_TEST,CountBenchQA,OCRBench,RealWorldQA,MathVerse_MINI,AI2D_TEST,ScienceQA_TEST,MMBench_DEV_EN_V11,MMStar,CV-Bench-2D,CV-Bench-3D}"
export LMUData="${LMUData:-/share/wulijun/liyu/LMUData/oda}"
export EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/eval}"
export EVAL_CACHE_ROOT="${EVAL_CACHE_ROOT:-${EVAL_OUTPUT_ROOT}/cache}"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy

export CLUSTER="${CLUSTER:-raise}"
export MODEL_NAME="${1:?Usage: bash run.sh <MODEL_PATH> [NUM_GPUS]}"
export NUM_GPUS="${2:-8}"
export MODEL_BASENAME="$(basename "${MODEL_NAME}")"
export EVAL_RUN_SUFFIX="${EVAL_RUN_SUFFIX:-}"
export EVAL_MODEL_ALIAS="${EVAL_MODEL_ALIAS:-${MODEL_BASENAME}${EVAL_RUN_SUFFIX}}"
EVAL_EXCLUDE_DEFAULT="SH-IDC1-10-140-37-10,SH-IDC1-10-140-37-49,SH-IDC1-10-140-37-137"
if [ "${EVAL_DISABLE_EXCLUDE:-0}" = "1" ] || [ "${EVAL_DISABLE_EXCLUDE:-}" = "true" ]; then
    export EVAL_EXCLUDE_NODES=""
else
    export EVAL_EXCLUDE_NODES="${EVAL_EXCLUDE_NODES:-${EVAL_EXCLUDE_DEFAULT}}"
fi
if [ -n "${EVAL_EXCLUDE_EXTRA:-}" ]; then
    EVAL_EXCLUDE_NODES="${EVAL_EXCLUDE_NODES},${EVAL_EXCLUDE_EXTRA}"
fi
VLLM_EXTRA_ARGS=()
if [ "${VLLM_ENFORCE_EAGER:-0}" = "1" ] || [ "${VLLM_ENFORCE_EAGER:-}" = "true" ]; then
    VLLM_EXTRA_ARGS+=(--enforce-eager)
fi
if [ "${VLLM_TRUST_REMOTE_CODE:-0}" = "1" ] || [ "${VLLM_TRUST_REMOTE_CODE:-}" = "true" ]; then
    VLLM_EXTRA_ARGS+=(--trust-remote-code)
fi
if [ "${#VLLM_EXTRA_ARGS[@]}" -gt 0 ]; then
    export VLLM_EXTRA_ARGS_STR="${VLLM_EXTRA_ARGS[*]}"
else
    export VLLM_EXTRA_ARGS_STR=""
fi

echo "MODEL_NAME: ${MODEL_NAME}"
echo "MODEL_BASENAME: ${MODEL_BASENAME}"
echo "EVAL_MODEL_ALIAS: ${EVAL_MODEL_ALIAS}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "DATASET: ${DATASET}"
echo "LMUData: ${LMUData}"
echo "EVAL_OUTPUT_ROOT: ${EVAL_OUTPUT_ROOT}"
echo "EVAL_CACHE_ROOT: ${EVAL_CACHE_ROOT}"
echo "EVAL_ENABLE_THINKING: ${EVAL_ENABLE_THINKING:-0}"
echo "EVAL_SYSTEM_PROMPT_MODE: ${EVAL_SYSTEM_PROMPT_MODE:-default}"
echo "EVAL_SYSTEM_PROMPT: ${EVAL_SYSTEM_PROMPT:-<unset>}"
echo "EVAL_TEMPERATURE: ${EVAL_TEMPERATURE:-0.0}"
echo "EVAL_TOP_P: ${EVAL_TOP_P:-0.95}"
echo "EVAL_MAX_TOKENS: ${EVAL_MAX_TOKENS:-4096}"
echo "EVAL_REQUEST_TIMEOUT: ${EVAL_REQUEST_TIMEOUT:-1800}"
echo "EVAL_MAX_CONCURRENT: ${EVAL_MAX_CONCURRENT:-250}"
echo "EVAL_REPETITION_PENALTY: ${EVAL_REPETITION_PENALTY:-1.05}"
echo "VLLM_MAX_MODEL_LEN: ${VLLM_MAX_MODEL_LEN:-32768}"
echo "VLLM_MAX_NUM_SEQS: ${VLLM_MAX_NUM_SEQS:-512}"
echo "VLLM_GPU_MEMORY_UTILIZATION: ${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
echo "VLLM_ENFORCE_EAGER: ${VLLM_ENFORCE_EAGER:-0}"
echo "VLLM_TRUST_REMOTE_CODE: ${VLLM_TRUST_REMOTE_CODE:-0}"
echo "VLLM_EXTRA_ARGS: ${VLLM_EXTRA_ARGS_STR:-<none>}"
echo "EVAL_EXCLUDE_NODES: ${EVAL_EXCLUDE_NODES}"
echo "CLUSTER: ${CLUSTER}"

mkdir -p server_logs distill_logs verify_logs "${EVAL_OUTPUT_ROOT}/outputs" "${EVAL_CACHE_ROOT}"

SRUN_CMD=(srun -p "${CLUSTER}")
if [ "${EVAL_SRUN_ASYNC:-0}" = "1" ] || [ "${EVAL_SRUN_ASYNC:-}" = "true" ]; then
    SRUN_CMD+=(--async)
fi

SRUN_EXCLUDE_ARGS=()
if [ -n "${EVAL_EXCLUDE_NODES}" ]; then
    SRUN_EXCLUDE_ARGS=(-x "${EVAL_EXCLUDE_NODES}")
fi

"${SRUN_CMD[@]}" --gres="gpu:${NUM_GPUS}" --job-name="eval_${EVAL_MODEL_ALIAS}" --quotatype="${EVAL_QUOTA:-auto}" \
    ${SRUN_EXCLUDE_ARGS[@]+"${SRUN_EXCLUDE_ARGS[@]}"} \
    bash -c '
        set -euo pipefail
        echo "Starting vLLM eval job"

        nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
        head_node=$(echo "$nodes" | head -1)
        head_ip=$(echo "$head_node" | grep -oE "[0-9]+-[0-9]+-[0-9]+-[0-9]+$" | tr "-" ".")
        echo "Head node IP: $head_ip (node: $head_node)"
        SERVER_PORT="${VLLM_PORT:-$((18000 + SLURM_JOB_ID % 20000))}"
        echo "vLLM port: ${SERVER_PORT}"

        VLLM_SIF=${VLLM_SIF:-/mnt/dhwfile/raise/user/linhonglin/apptainer/vllm-v0.17.1.sif}
        PATCH_LIB="${PATCH_LIB:-/mnt/dhwfile/raise/user/linhonglin/mmfinereason/eval/vllm_patch/lib}"
        mkdir -p "${EVAL_CACHE_ROOT}/hf" "${EVAL_CACHE_ROOT}/xdg/${EVAL_MODEL_ALIAS}" "${EVAL_CACHE_ROOT}/triton/${EVAL_MODEL_ALIAS}"
        : > server_logs/${EVAL_MODEL_ALIAS}.log

        apptainer exec --nv --cleanenv --bind /share:/share,/mnt:/mnt \
            --bind /mnt/dhwfile/liuzheng/mathrl/ChartDiff/eval/VLMEvalKit:/opt/VLMEvalKit \
            --env LMUData="${LMUData}" \
            --env HF_HOME="${EVAL_CACHE_ROOT}/hf" \
            --env HF_DATASETS_CACHE="${EVAL_CACHE_ROOT}/hf/datasets" \
            --env XDG_CACHE_HOME="${EVAL_CACHE_ROOT}/xdg/${EVAL_MODEL_ALIAS}" \
            --env TRITON_CACHE_DIR="${EVAL_CACHE_ROOT}/triton/${EVAL_MODEL_ALIAS}" \
            --env PYTHONPATH="$PATCH_LIB" \
            --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
            "$VLLM_SIF" \
            vllm serve "$MODEL_NAME" \
            --port "${SERVER_PORT}" \
            --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}" \
            --tensor-parallel-size "$NUM_GPUS" \
            --limit-mm-per-prompt.video 0 \
            --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.8}" \
            --enable-chunked-prefill \
            --max-num-seqs "${VLLM_MAX_NUM_SEQS:-512}" \
            ${VLLM_EXTRA_ARGS_STR:-} \
            2>&1 | tee server_logs/${EVAL_MODEL_ALIAS}.log &

        echo "Waiting for vLLM service to start..."
        TIMEOUT=${VLLM_STARTUP_TIMEOUT:-600}; ELAPSED=0; INTERVAL=20
        while [ $ELAPSED -lt $TIMEOUT ]; do
            if grep -q "Application startup complete." server_logs/${EVAL_MODEL_ALIAS}.log 2>/dev/null; then
                echo "vLLM ready!"
                break
            fi
            sleep $INTERVAL
            ELAPSED=$((ELAPSED + INTERVAL))
            echo "Waited ${ELAPSED}s..."
        done

        if [ $ELAPSED -ge $TIMEOUT ]; then
            echo "vLLM service failed to start within timeout"
            exit 1
        fi

        apptainer exec --nv --cleanenv --bind /share:/share,/mnt:/mnt \
            --bind /mnt/dhwfile/liuzheng/mathrl/ChartDiff/eval/VLMEvalKit:/opt/VLMEvalKit \
            --env LMUData="${LMUData}" \
            --env EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}" \
            --env EVAL_MODEL_ALIAS="${EVAL_MODEL_ALIAS}" \
            --env EVAL_ENABLE_THINKING="${EVAL_ENABLE_THINKING:-0}" \
            --env EVAL_SYSTEM_PROMPT_MODE="${EVAL_SYSTEM_PROMPT_MODE:-}" \
            --env EVAL_SYSTEM_PROMPT="${EVAL_SYSTEM_PROMPT:-}" \
            --env EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}" \
            --env EVAL_TOP_P="${EVAL_TOP_P:-0.95}" \
            --env EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-4096}" \
            --env EVAL_REQUEST_TIMEOUT="${EVAL_REQUEST_TIMEOUT:-1800}" \
            --env EVAL_REPETITION_PENALTY="${EVAL_REPETITION_PENALTY:-1.05}" \
            --env EVAL_IMAGE_TARGET_SIZES="${EVAL_IMAGE_TARGET_SIZES:--1,1024,768,512}" \
            --env HF_HOME="${EVAL_CACHE_ROOT}/hf" \
            --env HF_DATASETS_CACHE="${EVAL_CACHE_ROOT}/hf/datasets" \
            --env XDG_CACHE_HOME="${EVAL_CACHE_ROOT}/xdg/${EVAL_MODEL_ALIAS}" \
            --env TRITON_CACHE_DIR="${EVAL_CACHE_ROOT}/triton/${EVAL_MODEL_ALIAS}" \
            --env ALL_PROXY= --env all_proxy= \
            --env HTTP_PROXY= --env HTTPS_PROXY= \
            --env http_proxy= --env https_proxy= \
            /mnt/dhwfile/raise/user/caimengzhang/env/vmleval.sif \
            python run.py --datasets "$DATASET" --model_name "$MODEL_NAME" --url "${head_ip}:${SERVER_PORT}" --max_concurrent "${EVAL_MAX_CONCURRENT:-250}" \
                2>&1 | tee "distill_logs/${EVAL_MODEL_ALIAS}.log"
    '

if [ "${EVAL_SRUN_ASYNC:-0}" = "1" ] || [ "${EVAL_SRUN_ASYNC:-}" = "true" ]; then
    echo "Submitted async eval job for ${EVAL_MODEL_ALIAS}. Verify manually after rollout finishes:"
    echo "  bash verify.sh \"${EVAL_OUTPUT_ROOT}/outputs/${EVAL_MODEL_ALIAS}\""
    exit 0
fi

OUTPUT_PATH="${EVAL_OUTPUT_ROOT}/outputs/${EVAL_MODEL_ALIAS}"
bash verify.sh "${OUTPUT_PATH}"
