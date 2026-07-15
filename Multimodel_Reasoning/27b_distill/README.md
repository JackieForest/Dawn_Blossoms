# Qwen3.5-27B Distill Rollout

This directory mirrors `9b_distill` and runs Qwen3.5-27B rollout generation on the samples passed from the 9B all-wrong filter.

## Paths

- Input splits: `/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/qwen35_27b_distill/splits`
- Output root: `/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/qwen35_27b_distill`
- Success JSONL: `/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/qwen35_27b_distill/rollouts`
- Failed JSONL: `/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/qwen35_27b_distill/failed_rollouts`
- Stop flags: `/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/qwen35_27b_distill/stop_files`

## Run

```bash
cd /mnt/petrelfs/linjuekai/Multimodel_Reasoning/27b_distill
mkdir -p logs/qwen35_27b_distill
nohup bash scripts/launch.sh config/qwen35_27b_distill.yaml > logs/qwen35_27b_distill/launch_all.log 2>&1 &
```

## Monitor

```bash
cd /mnt/petrelfs/linjuekai/Multimodel_Reasoning/27b_distill
python3 scripts/status.py --config config/qwen35_27b_distill.yaml
squeue -u "$USER" | grep qwen35_27b_distill
```

## Stop

```bash
squeue -u "$USER" | awk '$3 ~ /_qwen35_27b_distill$/ {print $1}' | xargs -r scancel
pkill -f "single_job.sh config/qwen35_27b_distill.yaml" || true
```

Each split job requests 2 GPUs and starts vLLM with `tensor_parallel_size: 2`.
