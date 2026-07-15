#!/bin/bash
set -euo pipefail

PROJECT_DIR=/mnt/petrelfs/linjuekai/Multimodel_Reasoning/eval
RL_DIR=/mnt/petrelfs/linjuekai/Multimodel_Reasoning/rl
TRAIN_ROOT=/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/training/rl/qwen35_4b_full_sft_chart_table_doc_gspo_qwen_judge_strict_mixwrong_16k
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/eval}
EVAL_CACHE_ROOT=${EVAL_CACHE_ROOT:-${EVAL_OUTPUT_ROOT}/cache}
LMUData=${LMUData:-/share/wulijun/liyu/LMUData/oda}
VLLM_SIF=${VLLM_SIF:-/mnt/dhwfile/raise/user/linhonglin/apptainer/vllm-v0.17.1.sif}
RUN_SIF=${RUN_SIF:-/mnt/dhwfile/raise/user/caimengzhang/env/vmleval.sif}
PATCH_LIB=${PATCH_LIB:-/mnt/dhwfile/raise/user/linhonglin/mmfinereason/eval/vllm_patch/lib}
DATASET=${DATASET:-ChartQA_TEST,CharXiv_reasoning_val,EvoChart}
SUBMIT_SLEEP=${SUBMIT_SLEEP:-0.5}

mkdir -p "${PROJECT_DIR}/launcher_logs" "${PROJECT_DIR}/server_logs" "${PROJECT_DIR}/distill_logs" "${EVAL_OUTPUT_ROOT}/outputs" "${EVAL_CACHE_ROOT}"

submit_step() {
  local step="$1"
  local fix_script="${RL_DIR}/fix_chart_strict_mixwrong_step${step}_hf_vllm.py"
  local model_path="${TRAIN_ROOT}/global_step_${step}/actor/huggingface_vllm_fixed"
  local alias="qwen35_4b_chart_table_doc_strict_mixwrong_step${step}_fixed_chart3"
  local log_path="${PROJECT_DIR}/launcher_logs/${alias}.log"

  if [ ! -f "${fix_script}" ]; then
    echo "Missing fix script: ${fix_script}" >&2
    exit 1
  fi

  echo "Submitting ${alias} -> ${log_path}"
  srun -p raise --quotatype=spot --gres=gpu:1 --async --job-name="eval_chart_${step}" \
    bash -lc "
      set -euo pipefail
      cd '${PROJECT_DIR}'
      unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy

      echo 'Converting step ${step} to huggingface_vllm_fixed'
      apptainer exec --nv --cleanenv --bind /share:/share,/mnt:/mnt '${VLLM_SIF}' \
        python '${fix_script}'

      MODEL_NAME='${model_path}'
      EVAL_MODEL_ALIAS='${alias}'
      NUM_GPUS=1
      DATASET='${DATASET}'
      SERVER_PORT=\$((18000 + SLURM_JOB_ID % 20000))
      nodes=\$(scontrol show hostnames \"\$SLURM_JOB_NODELIST\")
      head_node=\$(echo \"\$nodes\" | head -1)
      head_ip=\$(echo \"\$head_node\" | grep -oE '[0-9]+-[0-9]+-[0-9]+-[0-9]+$' | tr '-' '.')

      echo \"MODEL_NAME: \${MODEL_NAME}\"
      echo \"EVAL_MODEL_ALIAS: \${EVAL_MODEL_ALIAS}\"
      echo \"DATASET: \${DATASET}\"
      echo \"Head node IP: \${head_ip}\"
      echo \"vLLM port: \${SERVER_PORT}\"

      mkdir -p '${PROJECT_DIR}/server_logs' '${PROJECT_DIR}/distill_logs' '${EVAL_CACHE_ROOT}/hf' '${EVAL_CACHE_ROOT}/xdg/${alias}' '${EVAL_CACHE_ROOT}/triton/${alias}'
      : > '${PROJECT_DIR}/server_logs/${alias}.log'

      apptainer exec --nv --cleanenv --bind /share:/share,/mnt:/mnt \
        --bind /mnt/dhwfile/liuzheng/mathrl/ChartDiff/eval/VLMEvalKit:/opt/VLMEvalKit \
        --env LMUData='${LMUData}' \
        --env HF_HOME='${EVAL_CACHE_ROOT}/hf' \
        --env HF_DATASETS_CACHE='${EVAL_CACHE_ROOT}/hf/datasets' \
        --env XDG_CACHE_HOME='${EVAL_CACHE_ROOT}/xdg/${alias}' \
        --env TRITON_CACHE_DIR='${EVAL_CACHE_ROOT}/triton/${alias}' \
        --env PYTHONPATH='${PATCH_LIB}' \
        --env CUDA_VISIBLE_DEVICES=\"\${CUDA_VISIBLE_DEVICES}\" \
        '${VLLM_SIF}' \
        vllm serve \"\${MODEL_NAME}\" \
        --port \"\${SERVER_PORT}\" \
        --max-model-len '${VLLM_MAX_MODEL_LEN:-32768}' \
        --tensor-parallel-size 1 \
        --limit-mm-per-prompt.video 0 \
        --gpu-memory-utilization '${VLLM_GPU_MEMORY_UTILIZATION:-0.8}' \
        --enable-chunked-prefill \
        --max-num-seqs '${VLLM_MAX_NUM_SEQS:-128}' \
        --trust-remote-code \
        2>&1 | tee '${PROJECT_DIR}/server_logs/${alias}.log' &

      echo 'Waiting for vLLM service to start...'
      TIMEOUT='${VLLM_STARTUP_TIMEOUT:-600}'; ELAPSED=0; INTERVAL=20
      while [ \$ELAPSED -lt \$TIMEOUT ]; do
        if grep -q 'Application startup complete.' '${PROJECT_DIR}/server_logs/${alias}.log' 2>/dev/null; then
          echo 'vLLM ready!'
          break
        fi
        sleep \$INTERVAL
        ELAPSED=\$((ELAPSED + INTERVAL))
        echo \"Waited \${ELAPSED}s...\"
      done
      if [ \$ELAPSED -ge \$TIMEOUT ]; then
        echo 'vLLM service failed to start within timeout'
        exit 1
      fi

      apptainer exec --nv --cleanenv --bind /share:/share,/mnt:/mnt \
        --bind /mnt/dhwfile/liuzheng/mathrl/ChartDiff/eval/VLMEvalKit:/opt/VLMEvalKit \
        --env LMUData='${LMUData}' \
        --env EVAL_OUTPUT_ROOT='${EVAL_OUTPUT_ROOT}' \
        --env EVAL_MODEL_ALIAS=\"\${EVAL_MODEL_ALIAS}\" \
        --env EVAL_ENABLE_THINKING='${EVAL_ENABLE_THINKING:-1}' \
        --env EVAL_SYSTEM_PROMPT_MODE='${EVAL_SYSTEM_PROMPT_MODE:-sft}' \
        --env EVAL_TEMPERATURE='${EVAL_TEMPERATURE:-0.0}' \
        --env EVAL_TOP_P='${EVAL_TOP_P:-0.95}' \
        --env EVAL_MAX_TOKENS='${EVAL_MAX_TOKENS:-8192}' \
        --env EVAL_REQUEST_TIMEOUT='${EVAL_REQUEST_TIMEOUT:-7200}' \
        --env EVAL_REPETITION_PENALTY='${EVAL_REPETITION_PENALTY:-1.05}' \
        --env EVAL_IMAGE_TARGET_SIZES='${EVAL_IMAGE_TARGET_SIZES:--1,1024,768,512}' \
        --env HF_HOME='${EVAL_CACHE_ROOT}/hf' \
        --env HF_DATASETS_CACHE='${EVAL_CACHE_ROOT}/hf/datasets' \
        --env XDG_CACHE_HOME='${EVAL_CACHE_ROOT}/xdg/${alias}' \
        --env TRITON_CACHE_DIR='${EVAL_CACHE_ROOT}/triton/${alias}' \
        --env ALL_PROXY= --env all_proxy= --env HTTP_PROXY= --env HTTPS_PROXY= \
        --env http_proxy= --env https_proxy= \
        '${RUN_SIF}' \
        python run.py --datasets \"\${DATASET}\" --model_name \"\${MODEL_NAME}\" --url \"\${head_ip}:\${SERVER_PORT}\" --max_concurrent '${EVAL_MAX_CONCURRENT:-64}' \
        2>&1 | tee '${PROJECT_DIR}/distill_logs/${alias}.log'
    " > "${log_path}" 2>&1

  sleep "${SUBMIT_SLEEP}"
}

submit_step 180
submit_step 220

echo "Submitted step180 and step220 convert+rollout jobs on raise spot."
