#!/bin/bash
# Roll out the logic-domain RL step70 checkpoint on VisuLogic and LogicVista.
#
# Usage:
#   bash script/launch_qwen35_4b_logic_rl_step70_rollout.sh
set -euo pipefail

PROJECT_DIR=/mnt/petrelfs/linjuekai/Multimodel_Reasoning/eval
MODEL_PATH=/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/training/rl/qwen35_4b_full_sft_logic_game_puzzle_gspo_compass/global_step_70/actor_hf_merged_science_style_rl_lmhead
ALIAS=qwen35_4b_logic_rl_step70_science_style

cd "${PROJECT_DIR}"
mkdir -p launcher_logs server_logs distill_logs

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Missing model path: ${MODEL_PATH}" >&2
  exit 1
fi

echo "Launching ${ALIAS}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATASET=VisuLogic,LogicVista"

CLUSTER="${CLUSTER:-sciverse_agent}" \
EVAL_SRUN_ASYNC=1 \
EVAL_QUOTA="${EVAL_QUOTA:-reserved}" \
DATASET="${DATASET:-VisuLogic,LogicVista}" \
EVAL_MODEL_ALIAS="${ALIAS}" \
EVAL_SYSTEM_PROMPT_MODE="${EVAL_SYSTEM_PROMPT_MODE:-sft}" \
EVAL_ENABLE_THINKING="${EVAL_ENABLE_THINKING:-1}" \
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-16384}" \
EVAL_REQUEST_TIMEOUT="${EVAL_REQUEST_TIMEOUT:-7200}" \
EVAL_MAX_CONCURRENT="${EVAL_MAX_CONCURRENT:-64}" \
EVAL_IMAGE_TARGET_SIZES="${EVAL_IMAGE_TARGET_SIZES:-1024,768,512}" \
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-50000}" \
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}" \
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}" \
VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}" \
nohup bash run.sh "${MODEL_PATH}" "${NUM_GPUS:-1}" > "launcher_logs/${ALIAS}.log" 2>&1 &

echo "Submitted rollout launcher."
echo "Launcher log: ${PROJECT_DIR}/launcher_logs/${ALIAS}.log"
echo "Server log:   ${PROJECT_DIR}/server_logs/${ALIAS}.log"
echo "Rollout log:  ${PROJECT_DIR}/distill_logs/${ALIAS}.log"
echo "Output:       /mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/eval/outputs/${ALIAS}/rollout.jsonl"
