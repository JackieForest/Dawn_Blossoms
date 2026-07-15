# Qwen3.5-9B Distillation Rollouts

Input split directory:

```text
/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/qwen35_9b_distill/splits
```

Output, cache, logs written under:

```text
/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/qwen35_9b_distill
```

Run all 400 splits:

```bash
cd /mnt/petrelfs/linjuekai/Multimodel_Reasoning/9b_distill
nohup bash scripts/launch.sh config/qwen35_9b_distill.yaml \
  > logs/qwen35_9b_distill/launch_all.log 2>&1 &
```

Check status:

```bash
python3 scripts/status.py --config config/qwen35_9b_distill.yaml
squeue -u "$USER" | grep qwen35_9b_distill
```

Stop all jobs:

```bash
pkill -9 -f 'launch\.sh config/qwen35_9b_distill\.yaml' || true
pkill -9 -f 'single_job\.sh config/qwen35_9b_distill\.yaml' || true
pkill -9 -f 'slurm_worker\.sh config/qwen35_9b_distill\.yaml' || true
pkill -9 -f 'phoenix-srun .*qwen35_9b_distill' || true

squeue -u "$USER" -h -o '%i %j' \
  | awk '$2 ~ /qwen35_9b_distill/ {print $1}' \
  | xargs -r scancel
```
