#!/bin/bash
set -uo pipefail

CONFIG="${1:?Usage: single_job.sh <config.yaml> <shard_id>}"
SHARD_ID="${2:?Usage: single_job.sh <config.yaml> <shard_id>}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

eval "$(python3 "${PROJECT_DIR}/config_env.py" --config "${CONFIG}")"

PARTITION="${SLURM_PARTITION:-sciverse_agent}"
QUOTA="${SLURM_QUOTA_TYPE:-spot}"
GPUS="${SLURM_GPUS:-1}"
TIME_LIMIT="${SLURM_TIME:-43200}"
EXCLUDE="${SLURM_EXCLUDE:-}"
EXP_NAME="${NAME:-qwen35_9b_best_of_n}"
OUTPUT="${OUTPUT_DIR:?missing output_dir}"
STOP_FILE="${OUTPUT}/stop_files/shard_${SHARD_ID}.flag"
JOB_NAME="${SHARD_ID}_${EXP_NAME}"

cleanup_old_jobs() {
    local old_jobs
    old_jobs=$(squeue -u "${USER}" -n "${JOB_NAME}" -o "%i" --noheader 2>/dev/null)
    if [[ -n "${old_jobs}" ]]; then
        echo "[${EXP_NAME}] cancelling old jobs for shard ${SHARD_ID}: ${old_jobs}"
        echo "${old_jobs}" | while read -r jid; do
            [[ -n "${jid}" ]] && scancel "${jid}" 2>/dev/null || true
        done
        sleep 5
    fi
}

while true; do
    if [[ -f "${STOP_FILE}" ]]; then
        echo "[${EXP_NAME}] shard ${SHARD_ID} stop flag exists; exiting."
        cleanup_old_jobs
        exit 0
    fi
    cleanup_old_jobs
    SRUN=(srun -p "${PARTITION}" --gres="gpu:${GPUS}" --quotatype="${QUOTA}" --job-name="${JOB_NAME}" --time="${TIME_LIMIT}")
    if [[ -n "${EXCLUDE}" ]]; then
        SRUN+=(-x "${EXCLUDE}")
    fi
    echo "[${EXP_NAME}] submitting shard ${SHARD_ID}: partition=${PARTITION} quota=${QUOTA} gpus=${GPUS}"
    "${SRUN[@]}" bash "${PROJECT_DIR}/slurm_worker.sh" "${CONFIG}" "${SHARD_ID}"
    rc=$?
    echo "[${EXP_NAME}] shard ${SHARD_ID} attempt exited rc=${rc}; retrying after 20s unless complete."
    sleep 20
done
