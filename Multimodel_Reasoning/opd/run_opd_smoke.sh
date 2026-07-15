#!/usr/bin/env bash
set -xeuo pipefail

OPD_DIR=${OPD_DIR:-/mnt/petrelfs/linjuekai/Multimodel_Reasoning/opd}
RL_DATA_DIR=${RL_DATA_DIR:-/mnt/petrelfs/linjuekai/Multimodel_Reasoning/rl_data/new_rl_data}
VERL_DIR=${VERL_DIR:-/mnt/dhwfile/raise/user/linjuekai/verl}
SIF=${SIF:-/mnt/dhwfile/raise/user/linhonglin/apptainer/vllm-v0.17.1.sif}
OVERLAY=${OVERLAY:-/mnt/dhwfile/raise/user/linhonglin/apptainer/verl_extra}
HF_HOME=${HF_HOME:-/mnt/dhwfile/raise/user/linhonglin/hf/huggingface}
USER_CACHE_DIR=${USER_CACHE_DIR:-/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/cache}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${USER_CACHE_DIR}/huggingface/datasets}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-${USER_CACHE_DIR}/xdg}

RAW_STUDENT_MODEL=${RAW_STUDENT_MODEL:-/mnt/dhwfile/raise/user/linjuekai/models/Qwen3.5-4B-Base}
TEACHER_MODEL=${TEACHER_MODEL:-/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/training/models/qwen35_4b_base_full_distill_sft_528k_16k}
STUDENT_MODEL=${STUDENT_MODEL:-${OPD_DIR}/models/Qwen3.5-4B-Base-chat-template}

USE_SMOKE_DATA=${USE_SMOKE_DATA:-1}
SMOKE_ROWS_PER_DOMAIN=${SMOKE_ROWS_PER_DOMAIN:-16}
SMOKE_DATA_DIR=${SMOKE_DATA_DIR:-${OPD_DIR}/data/smoke}

NNODES=${NNODES:-${SLURM_NNODES:-1}}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
TRAIN_GPUS=${TRAIN_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}

PROJECT_NAME=${PROJECT_NAME:-MMR_OPD}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen35_4b_base_to_full_sft_opd_smoke}
CKPTS_DIR=${CKPTS_DIR:-/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/training/opd/${EXPERIMENT_NAME}}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
MAX_NUM_TOKENS=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 ))
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
TRAIN_STEPS=${TRAIN_STEPS:-3}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:--1}
ACTOR_LR=${ACTOR_LR:-1e-6}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_N=${ROLLOUT_N:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.30}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
ROLLOUT_MM_PROCESSOR_CACHE_GB=${ROLLOUT_MM_PROCESSOR_CACHE_GB:-0}

TEACHER_NGPUS=${TEACHER_NGPUS:-1}
TEACHER_TP=${TEACHER_TP:-1}
TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.30}
DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k1}
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-True}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-32}

ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}

export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_opd_${SLURM_JOB_ID:-$$}}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export NO_PROXY="${NO_PROXY:-10.140.0.0/16,localhost,127.0.0.1}"
export no_proxy="${no_proxy:-10.140.0.0/16,localhost,127.0.0.1}"

mkdir -p "${OPD_DIR}/logs" "${CKPTS_DIR}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}" "${RAY_TMPDIR}"

if [ ! -f "${STUDENT_MODEL}/chat_template.jinja" ]; then
    python3 "${OPD_DIR}/prepare_student_with_chat_template.py" \
        --base-model "${RAW_STUDENT_MODEL}" \
        --chat-template "${TEACHER_MODEL}/chat_template.jinja" \
        --output-dir "${STUDENT_MODEL}"
fi

if [ "${USE_SMOKE_DATA}" = "1" ]; then
    python3 "${OPD_DIR}/prepare_opd_smoke_data.py" \
        --data-dir "${RL_DATA_DIR}" \
        --output-dir "${SMOKE_DATA_DIR}" \
        --rows-per-domain "${SMOKE_ROWS_PER_DOMAIN}"
    TRAIN_FILES="['${SMOKE_DATA_DIR}/train_chart_table_doc.parquet','${SMOKE_DATA_DIR}/train_logic_game_puzzle.parquet','${SMOKE_DATA_DIR}/train_math.parquet','${SMOKE_DATA_DIR}/train_science.parquet','${SMOKE_DATA_DIR}/train_spatial_general.parquet']"
else
    TRAIN_FILES="['${RL_DATA_DIR}/train_chart_table_doc.parquet','${RL_DATA_DIR}/train_logic_game_puzzle.parquet','${RL_DATA_DIR}/train_math.parquet','${RL_DATA_DIR}/train_science.parquet','${RL_DATA_DIR}/train_spatial_general.parquet']"
fi
VAL_FILES="${TRAIN_FILES}"

start_time=$(date +%Y%m%d)_$(date +%H%M%S)
log_file="${OPD_DIR}/logs/${EXPERIMENT_NAME}_${start_time}.log"

apptainer exec --nv --cleanenv \
    --env CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    --env PYTHONPATH="${OPD_DIR}:${OVERLAY}:${VERL_DIR}" \
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
    --env NO_PROXY="${NO_PROXY}" \
    --env no_proxy="${no_proxy}" \
    --bind /mnt:/mnt \
    "${SIF}" \
    bash -c "
    set -euo pipefail
    cd '${VERL_DIR}'
    ray stop --force 2>/dev/null || true
    sleep 2

    python -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        data.train_files=\"${TRAIN_FILES}\" \
        data.val_files=\"${VAL_FILES}\" \
        data.prompt_key=prompt \
        data.image_key=images \
        data.train_batch_size=${TRAIN_BATCH_SIZE} \
        data.max_prompt_length=${MAX_PROMPT_LENGTH} \
        data.max_response_length=${MAX_RESPONSE_LENGTH} \
        data.filter_overlong_prompts=True \
        data.filter_overlong_prompts_workers=4 \
        data.truncation=left \
        actor_rollout_ref.model.path='${STUDENT_MODEL}' \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
        actor_rollout_ref.actor.optim.total_training_steps=${TRAIN_STEPS} \
        actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_dynamic_bsz=False \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
        actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n=${ROLLOUT_N} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
        actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
        actor_rollout_ref.rollout.max_model_len=${MAX_NUM_TOKENS} \
        actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} \
        actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} \
        actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER} \
        actor_rollout_ref.rollout.calculate_log_probs=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
        +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=${ROLLOUT_MM_PROCESSOR_CACHE_GB} \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
        actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD} \
        actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
        critic.enable=False \
        reward.reward_model.enable=False \
        reward.reward_manager.name=naive \
        reward.custom_reward_function.path='${OPD_DIR}/zero_task_reward.py' \
        reward.custom_reward_function.name=compute_score \
        trainer.logger='[\"console\"]' \
        trainer.project_name='${PROJECT_NAME}' \
        trainer.experiment_name='${EXPERIMENT_NAME}' \
        trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
        trainer.nnodes=${NNODES} \
        trainer.val_before_train=False \
        trainer.test_freq=${TEST_FREQ} \
        trainer.save_freq=${SAVE_FREQ} \
        trainer.total_training_steps=${TRAIN_STEPS} \
        trainer.total_epochs=${TOTAL_EPOCHS} \
        trainer.default_local_dir='${CKPTS_DIR}' \
        trainer.resume_mode=disable \
        distillation.enabled=True \
        distillation.n_gpus_per_node=${TEACHER_NGPUS} \
        distillation.nnodes=1 \
        distillation.teacher_models.teacher_model.model_path='${TEACHER_MODEL}' \
        distillation.teacher_models.teacher_model.inference.name=vllm \
        distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP} \
        distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEMORY_UTILIZATION} \
        distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_NUM_TOKENS} \
        distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} \
        distillation.teacher_models.teacher_model.inference.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} \
        +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.max_logprobs=${DISTILLATION_TOPK} \
        distillation.distillation_loss.loss_mode=${DISTILLATION_LOSS_MODE} \
        distillation.distillation_loss.topk=${DISTILLATION_TOPK} \
        distillation.distillation_loss.use_task_rewards=False \
        distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRADIENT} \
        distillation.distillation_loss.loss_max_clamp=10.0 \
        distillation.distillation_loss.log_prob_min_clamp=-10.0
    " 2>&1 | tee "${log_file}"

echo "[run_opd_smoke] log: ${log_file}"
