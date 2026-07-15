#!/bin/bash
# Stop eval rollout/verify jobs and clean leftover vLLM servers.
# Usage:
#   bash stop_eval.sh [MODEL_BASENAME]
# Example:
#   bash stop_eval.sh Qwen3.5-4B

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJECT_DIR}"

MODEL_BASENAME="${1:-}"
USER_NAME="${USER}"

echo "== Cancel Slurm eval/verify jobs =="
if [ -n "${MODEL_BASENAME}" ]; then
    JOB_IDS=$(squeue -u "${USER_NAME}" -h -o "%i %j" \
        | awk -v model="${MODEL_BASENAME}" '
            $2 == "eval_" model || $2 == "verify_" model {print $1}
        ')
else
    JOB_IDS=$(squeue -u "${USER_NAME}" -h -o "%i %j" \
        | awk '$2 ~ /^(eval|verify)_/ {print $1}')
fi

if [ -n "${JOB_IDS:-}" ]; then
    echo "${JOB_IDS}" | xargs -r scancel
    echo "Cancelled jobs: ${JOB_IDS}"
else
    echo "No eval/verify Slurm jobs found."
fi

echo
echo "== Kill local launcher/wrapper processes =="
if [ -n "${MODEL_BASENAME}" ]; then
    LOCAL_PATTERNS=(
        "bash run.sh .*${MODEL_BASENAME}"
        "phoenix-srun .*job-name=(eval|verify)_.*${MODEL_BASENAME}"
        "srun .*job-name=(eval|verify)_.*${MODEL_BASENAME}"
    )
else
    LOCAL_PATTERNS=(
        "bash run.sh"
        "phoenix-srun .*job-name=(eval|verify)_"
        "srun .*job-name=(eval|verify)_"
    )
fi

for pat in "${LOCAL_PATTERNS[@]}"; do
    pids=$(pgrep -u "${USER_NAME}" -f "${pat}" || true)
    if [ -n "${pids}" ]; then
        echo "Killing local pattern: ${pat}"
        echo "${pids}" | xargs -r kill || true
    fi
done

sleep 2
for pat in "${LOCAL_PATTERNS[@]}"; do
    pids=$(pgrep -u "${USER_NAME}" -f "${pat}" || true)
    if [ -n "${pids}" ]; then
        echo "Force killing local pattern: ${pat}"
        echo "${pids}" | xargs -r kill -9 || true
    fi
done

echo
echo "Done. Check with:"
echo "  squeue -u ${USER_NAME} | rg 'eval_|verify_' || true"
echo "  tail -n 80 launcher_logs/*.log"
echo
echo "Note: this script intentionally does not kill remote tcp/8000 listeners,"
echo "because 4b_difficulty jobs may also use port 8000 on the same nodes."
