# Qwen3.5-122B Distill Rollout

Runs Qwen3.5-122B-A10B rollout generation for the 27B all-wrong fallback samples.

## Paths

- Input splits: `/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/qwen35_122b_distill/splits`
- Output root: `/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/qwen35_122b_distill`
- Success JSONL: `/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/qwen35_122b_distill/rollouts`
- Failed JSONL: `/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/qwen35_122b_distill/failed_rollouts`
- Stop flags: `/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/qwen35_122b_distill/stop_files`

## Run

```bash
cd /mnt/petrelfs/linjuekai/Multimodel_Reasoning/122b_distill
mkdir -p logs/qwen35_122b_distill
nohup bash scripts/launch.sh config/qwen35_122b_distill.yaml > logs/qwen35_122b_distill/launch_all.log 2>&1 &
```

## Monitor

```bash
cd /mnt/petrelfs/linjuekai/Multimodel_Reasoning/122b_distill
python3 scripts/status.py --config config/qwen35_122b_distill.yaml
squeue -u "$USER" | grep qwen35_122b_distill
```

## Stop

```bash
squeue -u "$USER" | awk '$3 ~ /_qwen35_122b_distill$/ {print $1}' | xargs -r scancel
pkill -f "single_job.sh config/qwen35_122b_distill.yaml" || true
```

Each split job requests 8 GPUs and starts vLLM with `tensor_parallel_size: 8`.
