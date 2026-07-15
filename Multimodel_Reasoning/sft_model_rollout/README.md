# Qwen3.5-4B Full SFT RL Candidate Rollouts

Input split directory:

```text
/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/training/rl_data/splits
```

Output, cache, and logs:

```text
/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/sft_model_rollout/output/qwen35_4b_full_sft_rl_rollout
/mnt/petrelfs/linjuekai/Multimodel_Reasoning/sft_model_rollout/logs/qwen35_4b_full_sft_rl_rollout
```

Model:

```text
/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/training/models/qwen35_4b_base_full_distill_sft_528k_16k
```

Run all 400 splits:

```bash
cd /mnt/petrelfs/linjuekai/Multimodel_Reasoning/sft_model_rollout
nohup bash scripts/launch.sh config/qwen35_4b_full_sft_rl_rollout.yaml   > logs/qwen35_4b_full_sft_rl_rollout/launch_all.log 2>&1 &
```

The launcher uses one GPU per split. It tries to keep the first 360 active jobs on reserved quota and submits the rest with spot quota.

Check status:

```bash
python3 scripts/status.py --config config/qwen35_4b_full_sft_rl_rollout.yaml
squeue -u "$USER" | grep qwen35_4b_full_sft_rl_rollout
```

Stop all jobs:

```bash
pkill -9 -f 'launch\.sh config/qwen35_4b_full_sft_rl_rollout\.yaml' || true
pkill -9 -f 'single_job\.sh config/qwen35_4b_full_sft_rl_rollout\.yaml' || true
pkill -9 -f 'slurm_worker\.sh config/qwen35_4b_full_sft_rl_rollout\.yaml' || true
pkill -9 -f 'srun .*qwen35_4b_full_sft_rl_rollout' || true

squeue -u "$USER" -h -o '%i %j'   | awk '$2 ~ /qwen35_4b_full_sft_rl_rollout/ {print $1}'   | xargs -r scancel
```
