#!/usr/bin/env bash
set -xeuo pipefail

DOMAIN=${1:-${DOMAIN:-science}}

RL_DIR=${RL_DIR:-/mnt/petrelfs/linjuekai/Multimodel_Reasoning/rl}
RL_DATA_DIR=${RL_DATA_DIR:-/mnt/petrelfs/linjuekai/Multimodel_Reasoning/rl_data/prompt_le_4096_pruned_sft_rollout_v4_drop_0125_700}
VERL_DIR=${VERL_DIR:-/mnt/dhwfile/raise/user/linjuekai/verl}
SIF=${SIF:-/mnt/dhwfile/raise/user/linhonglin/apptainer/vllm-v0.17.1.sif}
OVERLAY=${OVERLAY:-/mnt/dhwfile/raise/user/linhonglin/apptainer/verl_extra}
HF_HOME=${HF_HOME:-/mnt/dhwfile/raise/user/linhonglin/hf/huggingface}
USER_CACHE_DIR=${USER_CACHE_DIR:-/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/cache}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${USER_CACHE_DIR}/huggingface/datasets}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-${USER_CACHE_DIR}/xdg}

case "${DOMAIN}" in
    science) TRAIN_FILE=${TRAIN_FILE:-${RL_DATA_DIR}/train_science_compass_prompt_le_4096_pruned_sft_rollout.parquet} ;;
    chart_table_doc) TRAIN_FILE=${TRAIN_FILE:-${RL_DATA_DIR}/train_chart_table_doc_compass_prompt_le_4096_pruned_sft_rollout.parquet} ;;
    math) TRAIN_FILE=${TRAIN_FILE:-${RL_DATA_DIR}/train_math_compass_prompt_le_4096_pruned_sft_rollout.parquet} ;;
    logic_game_puzzle) TRAIN_FILE=${TRAIN_FILE:-${RL_DATA_DIR}/train_logic_game_puzzle_compass_prompt_le_4096_pruned_sft_rollout.parquet} ;;
    spatial_general) TRAIN_FILE=${TRAIN_FILE:-${RL_DATA_DIR}/train_spatial_general_compass_prompt_le_4096_pruned_sft_rollout.parquet} ;;
    *) echo "Unknown DOMAIN=${DOMAIN}" >&2; exit 2 ;;
esac
VAL_FILE=${VAL_FILE:-${TRAIN_FILE}}

MODEL_PATH=${MODEL_PATH:-/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/training/models/qwen35_4b_base_full_distill_sft_528k_16k}
PROJECT_NAME=${PROJECT_NAME:-MMR_RL_GSPO}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen35_4b_full_sft_${DOMAIN}_gspo_compass}
CKPTS_DIR=${CKPTS_DIR:-/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/training/rl/${EXPERIMENT_NAME}}

TRAIN_GPUS=${TRAIN_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}
VERIFY_GPUS=${VERIFY_GPUS:-0,1,2,3}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
NNODES=${NNODES:-${SLURM_NNODES:-1}}
RAY_PORT=${RAY_PORT:-$((25000 + (${SLURM_JOB_ID:-0} % 10000)))}
PORT=${PORT:-$((18765 + (${SLURM_JOB_ID:-0} % 1000)))}
URL_FILE=${URL_FILE:-${RL_DIR}/compass_urls/${DOMAIN}_${SLURM_JOB_ID:-manual}.txt}
START_COMPASS=${START_COMPASS:-1}
SERVE_VERIFIER_SCRIPT=${SERVE_VERIFIER_SCRIPT:-${RL_DIR}/serve_compass.sh}
VERIFIER_LOG_PREFIX=${VERIFIER_LOG_PREFIX:-serve_compass}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
ACTOR_SEQUENCE_PARALLEL_SIZE=${ACTOR_SEQUENCE_PARALLEL_SIZE:-2}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-32768}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-24576}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-256}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
ROLLOUT_MM_PROCESSOR_CACHE_GB=${ROLLOUT_MM_PROCESSOR_CACHE_GB:-0}
OVERLONG_BUFFER_ENABLE=${OVERLONG_BUFFER_ENABLE:-False}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-1024}
OVERLONG_BUFFER_PENALTY_FACTOR=${OVERLONG_BUFFER_PENALTY_FACTOR:-1.0}
OVERLONG_BUFFER_LOG=${OVERLONG_BUFFER_LOG:-False}
OVERLONG_BUFFER_CLAMP_MIN=${OVERLONG_BUFFER_CLAMP_MIN:-}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
ACTOR_LR=${ACTOR_LR:-1e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
WARMUP_STEPS=${WARMUP_STEPS:-10}
TRAIN_NCCL_TIMEOUT=${TRAIN_NCCL_TIMEOUT:-5400}
TRAIN_STEPS=${TRAIN_STEPS:-300}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:--1}
CHECKPOINT_SAVE_CONTENTS=${CHECKPOINT_SAVE_CONTENTS:-'["model","optimizer","extra","hf_model"]'}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-3e-4}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-4e-4}
TRAIN_LOGGER=${TRAIN_LOGGER:-'["console"]'}
REWARD_FUNCTION_PATH=${REWARD_FUNCTION_PATH:-${RL_DIR}/compass_format_reward.py}
REWARD_FUNCTION_NAME=${REWARD_FUNCTION_NAME:-compute_score}
CORRECTNESS_WEIGHT=${CORRECTNESS_WEIGHT:-0.9}
FORMAT_WEIGHT=${FORMAT_WEIGHT:-0.1}
RESUME_MODE=${RESUME_MODE:-disable}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-}

export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_${SLURM_JOB_ID:-$$}}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export NO_PROXY="${NO_PROXY:-10.140.0.0/16,localhost,127.0.0.1}"
export no_proxy="${no_proxy:-10.140.0.0/16,localhost,127.0.0.1}"
mkdir -p "${RAY_TMPDIR}" "${RL_DIR}/logs" "${RL_DIR}/compass_urls" "${CKPTS_DIR}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}"
OVERLONG_CLAMP_ARG=""
if [ -n "${OVERLONG_BUFFER_CLAMP_MIN}" ]; then
    OVERLONG_CLAMP_ARG="+reward.reward_kwargs.overlong_buffer_cfg.clamp_min=${OVERLONG_BUFFER_CLAMP_MIN}"
fi
if [ "${START_COMPASS}" = "1" ]; then
    echo "[run_gspo] START_COMPASS=1 uses VERIFY_GPUS=${VERIFY_GPUS}. Make sure these GPUs are not also used by TRAIN_GPUS=${TRAIN_GPUS}."
    rm -f "${URL_FILE}"
else
    mkdir -p "$(dirname "${URL_FILE}")"
fi

RAY_SRUN_PIDS=()
cleanup() {
    if [ -n "${COMPASS_PID:-}" ] && kill -0 "${COMPASS_PID}" 2>/dev/null; then
        kill "${COMPASS_PID}" 2>/dev/null || true
        wait "${COMPASS_PID}" 2>/dev/null || true
    fi
    if [ "${#RAY_SRUN_PIDS[@]}" -gt 0 ]; then
        for pid in "${RAY_SRUN_PIDS[@]}"; do
            kill "${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
        done
    fi
}
trap cleanup EXIT

if [ "${START_COMPASS}" = "1" ]; then
    CUDA_VISIBLE_DEVICES="${VERIFY_GPUS}" \
    PORT="${PORT}" TP="${COMPASS_TP:-4}" DP="${COMPASS_DP:-1}" URL_FILE="${URL_FILE}" \
    RL_DIR="${RL_DIR}" SIF="${SIF}" OVERLAY="${OVERLAY}" HF_HOME="${HF_HOME}" \
    bash "${SERVE_VERIFIER_SCRIPT}" > "${RL_DIR}/logs/${VERIFIER_LOG_PREFIX}_${DOMAIN}_${SLURM_JOB_ID:-manual}.log" 2>&1 &
    COMPASS_PID=$!
fi

for i in $(seq 1 120); do
    if [ -s "${URL_FILE}" ]; then
        COMPASS_VERIFIER_URL=$(cat "${URL_FILE}")
        if curl --noproxy "*" -fsS "${COMPASS_VERIFIER_URL}/models" >/dev/null 2>&1; then
            break
        fi
    fi
    echo "[run_gspo] waiting for CompassVerifier (${i}/120)..."
    sleep 10
done

COMPASS_VERIFIER_URL=$(cat "${URL_FILE}" 2>/dev/null || true)
if [ -z "${COMPASS_VERIFIER_URL}" ]; then
    echo "[run_gspo] CompassVerifier URL file is empty: ${URL_FILE}" >&2
    exit 3
fi
echo "[run_gspo] COMPASS_VERIFIER_URL=${COMPASS_VERIFIER_URL}"
COMPASS_VERIFIER_MODEL=${COMPASS_VERIFIER_MODEL:-opencompass/CompassVerifier-7B}
COMPASS_VERIFIER_TIMEOUT=${COMPASS_VERIFIER_TIMEOUT:-60}
COMPASS_VERIFIER_MAX_TOKENS=${COMPASS_VERIFIER_MAX_TOKENS:-2048}
COMPASS_VERIFIER_TOKENIZER_PATH=${COMPASS_VERIFIER_TOKENIZER_PATH:-/mnt/dhwfile/raise/user/linhonglin/hf/huggingface/hub/models--opencompass--CompassVerifier-7B/snapshots/676c83e3c62c199e0d6ad29cd31b6064c8d500a0}
COMPASS_VERIFIER_INPUT_MAX_TOKENS=${COMPASS_VERIFIER_INPUT_MAX_TOKENS:-30000}
COMPASS_CANDIDATE_TAIL_TOKENS=${COMPASS_CANDIDATE_TAIL_TOKENS:-30000}
COMPASS_CANDIDATE_MAX_CHARS=${COMPASS_CANDIDATE_MAX_CHARS:-120000}
COMPASS_CANDIDATE_FALLBACK_CHARS=${COMPASS_CANDIDATE_FALLBACK_CHARS:-60000}
QWEN_JUDGE_MODEL=${QWEN_JUDGE_MODEL:-${COMPASS_VERIFIER_MODEL}}
QWEN_JUDGE_TIMEOUT=${QWEN_JUDGE_TIMEOUT:-${COMPASS_VERIFIER_TIMEOUT}}
QWEN_JUDGE_TEMPERATURE=${QWEN_JUDGE_TEMPERATURE:-1.0}
QWEN_JUDGE_TOP_P=${QWEN_JUDGE_TOP_P:-1.0}
QWEN_JUDGE_MAX_TOKENS=${QWEN_JUDGE_MAX_TOKENS:-${COMPASS_VERIFIER_MAX_TOKENS}}
QWEN_JUDGE_RETRIES=${QWEN_JUDGE_RETRIES:-2}
QWEN_JUDGE_TOKENIZER_PATH=${QWEN_JUDGE_TOKENIZER_PATH:-${COMPASS_VERIFIER_TOKENIZER_PATH}}
QWEN_JUDGE_INPUT_MAX_TOKENS=${QWEN_JUDGE_INPUT_MAX_TOKENS:-${COMPASS_VERIFIER_INPUT_MAX_TOKENS}}
QWEN_JUDGE_CANDIDATE_TAIL_TOKENS=${QWEN_JUDGE_CANDIDATE_TAIL_TOKENS:-${COMPASS_CANDIDATE_TAIL_TOKENS}}
QWEN_JUDGE_CANDIDATE_MAX_CHARS=${QWEN_JUDGE_CANDIDATE_MAX_CHARS:-${COMPASS_CANDIDATE_MAX_CHARS}}
QWEN_JUDGE_CANDIDATE_FALLBACK_CHARS=${QWEN_JUDGE_CANDIDATE_FALLBACK_CHARS:-${COMPASS_CANDIDATE_FALLBACK_CHARS}}

RAY_ADDRESS_OVERRIDE=""
if [ "${NNODES}" -gt 1 ]; then
    mapfile -t SLURM_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    HEAD_NODE="${SLURM_NODES[0]}"
    HEAD_IP=$(echo "${HEAD_NODE}" | grep -oE '[0-9]+-[0-9]+-[0-9]+-[0-9]+$' | tr '-' '.')
    if [ -z "${HEAD_IP}" ]; then
        HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname -I | awk '{print $1}')
    fi
    RAY_ADDRESS_OVERRIDE="+ray_kwargs.ray_init.address=auto"
    echo "[run_gspo] starting Ray cluster: head=${HEAD_NODE} ip=${HEAD_IP} port=${RAY_PORT} nnodes=${NNODES}"

    for node in "${SLURM_NODES[@]}"; do
        srun --nodes=1 --ntasks=1 -w "${node}" \
            apptainer exec --nv --cleanenv \
            --env CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
            --env PYTHONPATH="${RL_DIR}:${OVERLAY}:${VERL_DIR}" \
            --env RAY_TMPDIR="${RAY_TMPDIR}" \
            --env HF_HOME="${HF_HOME}" \
            --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
            --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
            --env TRANSFORMERS_CACHE="${HF_HOME}/hub" \
            --env HF_HUB_OFFLINE=1 \
            --env NO_PROXY="10.140.0.0/16,localhost,127.0.0.1" \
            --env no_proxy="10.140.0.0/16,localhost,127.0.0.1" \
            --bind /mnt:/mnt \
            "${SIF}" ray stop --force >/dev/null 2>&1 || true
    done

    srun --exclusive --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
        apptainer exec --nv --cleanenv \
        --env CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
        --env PYTHONPATH="${RL_DIR}:${OVERLAY}:${VERL_DIR}" \
        --env RAY_TMPDIR="${RAY_TMPDIR}" \
        --env HF_HOME="${HF_HOME}" \
        --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
        --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
        --env TRANSFORMERS_CACHE="${HF_HOME}/hub" \
        --env HF_HUB_OFFLINE=1 \
        --env COMPASS_VERIFIER_URL="${COMPASS_VERIFIER_URL}" \
        --env COMPASS_VERIFIER_MODEL="${COMPASS_VERIFIER_MODEL}" \
        --env COMPASS_VERIFIER_TIMEOUT="${COMPASS_VERIFIER_TIMEOUT}" \
        --env COMPASS_VERIFIER_MAX_TOKENS="${COMPASS_VERIFIER_MAX_TOKENS}" \
        --env COMPASS_VERIFIER_TOKENIZER_PATH="${COMPASS_VERIFIER_TOKENIZER_PATH}" \
        --env COMPASS_VERIFIER_INPUT_MAX_TOKENS="${COMPASS_VERIFIER_INPUT_MAX_TOKENS}" \
        --env COMPASS_CANDIDATE_TAIL_TOKENS="${COMPASS_CANDIDATE_TAIL_TOKENS}" \
        --env COMPASS_CANDIDATE_MAX_CHARS="${COMPASS_CANDIDATE_MAX_CHARS}" \
        --env COMPASS_CANDIDATE_FALLBACK_CHARS="${COMPASS_CANDIDATE_FALLBACK_CHARS}" \
        --env QWEN_JUDGE_URL="${COMPASS_VERIFIER_URL}" \
        --env QWEN_JUDGE_MODEL="${QWEN_JUDGE_MODEL}" \
        --env QWEN_JUDGE_TIMEOUT="${QWEN_JUDGE_TIMEOUT}" \
        --env QWEN_JUDGE_TEMPERATURE="${QWEN_JUDGE_TEMPERATURE}" \
        --env QWEN_JUDGE_TOP_P="${QWEN_JUDGE_TOP_P}" \
        --env QWEN_JUDGE_MAX_TOKENS="${QWEN_JUDGE_MAX_TOKENS}" \
        --env QWEN_JUDGE_RETRIES="${QWEN_JUDGE_RETRIES}" \
        --env QWEN_JUDGE_TOKENIZER_PATH="${QWEN_JUDGE_TOKENIZER_PATH}" \
        --env QWEN_JUDGE_INPUT_MAX_TOKENS="${QWEN_JUDGE_INPUT_MAX_TOKENS}" \
        --env QWEN_JUDGE_CANDIDATE_TAIL_TOKENS="${QWEN_JUDGE_CANDIDATE_TAIL_TOKENS}" \
        --env QWEN_JUDGE_CANDIDATE_MAX_CHARS="${QWEN_JUDGE_CANDIDATE_MAX_CHARS}" \
        --env QWEN_JUDGE_CANDIDATE_FALLBACK_CHARS="${QWEN_JUDGE_CANDIDATE_FALLBACK_CHARS}" \
        --env NO_PROXY="10.140.0.0/16,localhost,127.0.0.1" \
        --env no_proxy="10.140.0.0/16,localhost,127.0.0.1" \
        --bind /mnt:/mnt \
        "${SIF}" ray start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" --num-gpus="${N_GPUS_PER_NODE}" --disable-usage-stats --block \
        > "${RL_DIR}/logs/ray_head_${DOMAIN}_${SLURM_JOB_ID:-manual}.log" 2>&1 &
    RAY_SRUN_PIDS+=("$!")
    sleep 15

    for node in "${SLURM_NODES[@]:1}"; do
        srun --exclusive --nodes=1 --ntasks=1 -w "${node}" \
            apptainer exec --nv --cleanenv \
            --env CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
            --env PYTHONPATH="${RL_DIR}:${OVERLAY}:${VERL_DIR}" \
            --env RAY_TMPDIR="${RAY_TMPDIR}" \
            --env HF_HOME="${HF_HOME}" \
            --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
            --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
            --env TRANSFORMERS_CACHE="${HF_HOME}/hub" \
            --env HF_HUB_OFFLINE=1 \
            --env COMPASS_VERIFIER_URL="${COMPASS_VERIFIER_URL}" \
            --env COMPASS_VERIFIER_MODEL="${COMPASS_VERIFIER_MODEL}" \
            --env COMPASS_VERIFIER_TIMEOUT="${COMPASS_VERIFIER_TIMEOUT}" \
            --env COMPASS_VERIFIER_MAX_TOKENS="${COMPASS_VERIFIER_MAX_TOKENS}" \
            --env COMPASS_VERIFIER_TOKENIZER_PATH="${COMPASS_VERIFIER_TOKENIZER_PATH}" \
            --env COMPASS_VERIFIER_INPUT_MAX_TOKENS="${COMPASS_VERIFIER_INPUT_MAX_TOKENS}" \
            --env COMPASS_CANDIDATE_TAIL_TOKENS="${COMPASS_CANDIDATE_TAIL_TOKENS}" \
            --env COMPASS_CANDIDATE_MAX_CHARS="${COMPASS_CANDIDATE_MAX_CHARS}" \
            --env COMPASS_CANDIDATE_FALLBACK_CHARS="${COMPASS_CANDIDATE_FALLBACK_CHARS}" \
            --env QWEN_JUDGE_URL="${COMPASS_VERIFIER_URL}" \
            --env QWEN_JUDGE_MODEL="${QWEN_JUDGE_MODEL}" \
            --env QWEN_JUDGE_TIMEOUT="${QWEN_JUDGE_TIMEOUT}" \
            --env QWEN_JUDGE_TEMPERATURE="${QWEN_JUDGE_TEMPERATURE}" \
            --env QWEN_JUDGE_TOP_P="${QWEN_JUDGE_TOP_P}" \
            --env QWEN_JUDGE_MAX_TOKENS="${QWEN_JUDGE_MAX_TOKENS}" \
            --env QWEN_JUDGE_RETRIES="${QWEN_JUDGE_RETRIES}" \
            --env QWEN_JUDGE_TOKENIZER_PATH="${QWEN_JUDGE_TOKENIZER_PATH}" \
            --env QWEN_JUDGE_INPUT_MAX_TOKENS="${QWEN_JUDGE_INPUT_MAX_TOKENS}" \
            --env QWEN_JUDGE_CANDIDATE_TAIL_TOKENS="${QWEN_JUDGE_CANDIDATE_TAIL_TOKENS}" \
            --env QWEN_JUDGE_CANDIDATE_MAX_CHARS="${QWEN_JUDGE_CANDIDATE_MAX_CHARS}" \
            --env QWEN_JUDGE_CANDIDATE_FALLBACK_CHARS="${QWEN_JUDGE_CANDIDATE_FALLBACK_CHARS}" \
            --env NO_PROXY="10.140.0.0/16,localhost,127.0.0.1" \
            --env no_proxy="10.140.0.0/16,localhost,127.0.0.1" \
            --bind /mnt:/mnt \
            "${SIF}" ray start --address="${HEAD_IP}:${RAY_PORT}" --num-gpus="${N_GPUS_PER_NODE}" --disable-usage-stats --block \
            > "${RL_DIR}/logs/ray_worker_${DOMAIN}_${SLURM_JOB_ID:-manual}_${node}.log" 2>&1 &
        RAY_SRUN_PIDS+=("$!")
    done
    sleep 20
fi

apptainer exec --nv --cleanenv \
    --env CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    --env PYTHONPATH="${RL_DIR}:${OVERLAY}:${VERL_DIR}" \
    --env RAY_TMPDIR="${RAY_TMPDIR}" \
    --env HF_HOME="${HF_HOME}" \
    --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
    --env TRANSFORMERS_CACHE="${HF_HOME}/hub" \
    --env HF_HUB_OFFLINE=1 \
    --env WANDB_MODE="${WANDB_MODE:-offline}" \
    --env WANDB_PROJECT="${PROJECT_NAME}" \
    --env WANDB_RUN_NAME="${EXPERIMENT_NAME}" \
    --env TOKENIZERS_PARALLELISM=true \
    --env NCCL_DEBUG=WARN \
    --env PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" \
    --env COMPASS_VERIFIER_URL="${COMPASS_VERIFIER_URL}" \
    --env COMPASS_VERIFIER_MODEL="${COMPASS_VERIFIER_MODEL}" \
    --env COMPASS_VERIFIER_TIMEOUT="${COMPASS_VERIFIER_TIMEOUT}" \
    --env COMPASS_VERIFIER_MAX_TOKENS="${COMPASS_VERIFIER_MAX_TOKENS}" \
    --env COMPASS_VERIFIER_TOKENIZER_PATH="${COMPASS_VERIFIER_TOKENIZER_PATH}" \
    --env COMPASS_VERIFIER_INPUT_MAX_TOKENS="${COMPASS_VERIFIER_INPUT_MAX_TOKENS}" \
    --env COMPASS_CANDIDATE_TAIL_TOKENS="${COMPASS_CANDIDATE_TAIL_TOKENS}" \
    --env COMPASS_CANDIDATE_MAX_CHARS="${COMPASS_CANDIDATE_MAX_CHARS}" \
    --env COMPASS_CANDIDATE_FALLBACK_CHARS="${COMPASS_CANDIDATE_FALLBACK_CHARS}" \
    --env QWEN_JUDGE_URL="${COMPASS_VERIFIER_URL}" \
    --env QWEN_JUDGE_MODEL="${QWEN_JUDGE_MODEL}" \
    --env QWEN_JUDGE_TIMEOUT="${QWEN_JUDGE_TIMEOUT}" \
    --env QWEN_JUDGE_TEMPERATURE="${QWEN_JUDGE_TEMPERATURE}" \
    --env QWEN_JUDGE_TOP_P="${QWEN_JUDGE_TOP_P}" \
    --env QWEN_JUDGE_MAX_TOKENS="${QWEN_JUDGE_MAX_TOKENS}" \
    --env QWEN_JUDGE_RETRIES="${QWEN_JUDGE_RETRIES}" \
    --env QWEN_JUDGE_TOKENIZER_PATH="${QWEN_JUDGE_TOKENIZER_PATH}" \
    --env QWEN_JUDGE_INPUT_MAX_TOKENS="${QWEN_JUDGE_INPUT_MAX_TOKENS}" \
    --env QWEN_JUDGE_CANDIDATE_TAIL_TOKENS="${QWEN_JUDGE_CANDIDATE_TAIL_TOKENS}" \
    --env QWEN_JUDGE_CANDIDATE_MAX_CHARS="${QWEN_JUDGE_CANDIDATE_MAX_CHARS}" \
    --env QWEN_JUDGE_CANDIDATE_FALLBACK_CHARS="${QWEN_JUDGE_CANDIDATE_FALLBACK_CHARS}" \
    --env NO_PROXY="10.140.0.0/16,localhost,127.0.0.1" \
    --env no_proxy="10.140.0.0/16,localhost,127.0.0.1" \
    --bind /mnt:/mnt \
    "${SIF}" \
    bash -c "
    set -euo pipefail
    cd '${VERL_DIR}'
    if [ '${NNODES}' -le 1 ]; then
        ray stop --force 2>/dev/null || true
        sleep 2
    fi

    python -m verl.trainer.main_ppo \
        ${RAY_ADDRESS_OVERRIDE} \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        actor_rollout_ref.nccl_timeout=${TRAIN_NCCL_TIMEOUT} \
        data.train_files='${TRAIN_FILE}' \
        data.val_files='${VAL_FILE}' \
        data.prompt_key=prompt \
        data.image_key=images \
        data.truncation=left \
        data.max_prompt_length=${MAX_PROMPT_LENGTH} \
        data.max_response_length=${MAX_RESPONSE_LENGTH} \
        data.train_batch_size=${TRAIN_BATCH_SIZE} \
        data.filter_overlong_prompts=True \
        data.filter_overlong_prompts_workers=8 \
        actor_rollout_ref.model.path='${MODEL_PATH}' \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
        actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
        actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
        actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
        actor_rollout_ref.actor.optim.lr_warmup_steps=${WARMUP_STEPS} \
        actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
        actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY} \
        actor_rollout_ref.actor.optim.total_training_steps=${TRAIN_STEPS} \
        actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ACTOR_SEQUENCE_PARALLEL_SIZE} \
        actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n=${ROLLOUT_N} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
        actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
        actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN} \
        actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER} \
        +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=${ROLLOUT_MM_PROCESSOR_CACHE_GB} \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} \
        actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} \
        actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
        actor_rollout_ref.rollout.top_p=${TOP_P} \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
        actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD} \
        reward.reward_manager.name=dapo \
        reward.custom_reward_function.path='${REWARD_FUNCTION_PATH}' \
        reward.custom_reward_function.name=${REWARD_FUNCTION_NAME} \
        +reward.custom_reward_function.reward_kwargs.correctness_weight=${CORRECTNESS_WEIGHT} \
        +reward.custom_reward_function.reward_kwargs.format_weight=${FORMAT_WEIGHT} \
        +reward.custom_reward_function.reward_kwargs.compass_verifier_url='${COMPASS_VERIFIER_URL}' \
        +reward.custom_reward_function.reward_kwargs.compass_verifier_model='${COMPASS_VERIFIER_MODEL}' \
        +reward.custom_reward_function.reward_kwargs.compass_verifier_timeout=${COMPASS_VERIFIER_TIMEOUT} \
        +reward.custom_reward_function.reward_kwargs.compass_verifier_max_tokens=${COMPASS_VERIFIER_MAX_TOKENS} \
        +reward.custom_reward_function.reward_kwargs.compass_verifier_tokenizer_path='${COMPASS_VERIFIER_TOKENIZER_PATH}' \
        +reward.custom_reward_function.reward_kwargs.compass_verifier_input_max_tokens=${COMPASS_VERIFIER_INPUT_MAX_TOKENS} \
        +reward.custom_reward_function.reward_kwargs.compass_candidate_tail_tokens=${COMPASS_CANDIDATE_TAIL_TOKENS} \
        +reward.custom_reward_function.reward_kwargs.compass_candidate_max_chars=${COMPASS_CANDIDATE_MAX_CHARS} \
        +reward.custom_reward_function.reward_kwargs.compass_candidate_fallback_chars=${COMPASS_CANDIDATE_FALLBACK_CHARS} \
        +reward.custom_reward_function.reward_kwargs.qwen_judge_url='${COMPASS_VERIFIER_URL}' \
        +reward.custom_reward_function.reward_kwargs.qwen_judge_model='${QWEN_JUDGE_MODEL}' \
        +reward.custom_reward_function.reward_kwargs.qwen_judge_timeout=${QWEN_JUDGE_TIMEOUT} \
        +reward.custom_reward_function.reward_kwargs.qwen_judge_temperature=${QWEN_JUDGE_TEMPERATURE} \
        +reward.custom_reward_function.reward_kwargs.qwen_judge_top_p=${QWEN_JUDGE_TOP_P} \
        +reward.custom_reward_function.reward_kwargs.qwen_judge_max_tokens=${QWEN_JUDGE_MAX_TOKENS} \
        +reward.custom_reward_function.reward_kwargs.qwen_judge_retries=${QWEN_JUDGE_RETRIES} \
        +reward.custom_reward_function.reward_kwargs.qwen_judge_tokenizer_path='${QWEN_JUDGE_TOKENIZER_PATH}' \
        +reward.custom_reward_function.reward_kwargs.qwen_judge_input_max_tokens=${QWEN_JUDGE_INPUT_MAX_TOKENS} \
        +reward.custom_reward_function.reward_kwargs.qwen_judge_candidate_tail_tokens=${QWEN_JUDGE_CANDIDATE_TAIL_TOKENS} \
        +reward.custom_reward_function.reward_kwargs.qwen_judge_candidate_max_chars=${QWEN_JUDGE_CANDIDATE_MAX_CHARS} \
        +reward.custom_reward_function.reward_kwargs.qwen_judge_candidate_fallback_chars=${QWEN_JUDGE_CANDIDATE_FALLBACK_CHARS} \
        +reward.reward_kwargs.overlong_buffer_cfg.enable=${OVERLONG_BUFFER_ENABLE} \
        +reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN} \
        +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_BUFFER_PENALTY_FACTOR} \
        +reward.reward_kwargs.overlong_buffer_cfg.log=${OVERLONG_BUFFER_LOG} \
        ${OVERLONG_CLAMP_ARG} \
        +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH} \
        trainer.logger='${TRAIN_LOGGER}' \
        trainer.project_name='${PROJECT_NAME}' \
        trainer.experiment_name='${EXPERIMENT_NAME}' \
        trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
        trainer.nnodes=${NNODES} \
        trainer.val_before_train=False \
        trainer.test_freq=${TEST_FREQ} \
        trainer.save_freq=${SAVE_FREQ} \
        actor_rollout_ref.actor.checkpoint.save_contents='${CHECKPOINT_SAVE_CONTENTS}' \
        trainer.total_training_steps=${TRAIN_STEPS} \
        trainer.total_epochs=${TOTAL_EPOCHS} \
        trainer.default_local_dir='${CKPTS_DIR}' \
        trainer.rollout_data_dir='${CKPTS_DIR}/rollout_data' \
        trainer.resume_mode=${RESUME_MODE} \
        trainer.resume_from_path='${RESUME_FROM_PATH}'
"
