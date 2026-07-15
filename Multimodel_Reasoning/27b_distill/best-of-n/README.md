# 27B Answer Likelihood Best-of-N

This project selects one best rollout for each 27B-correct sample.

Input:

- Splits: `/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/distill_results/27b_distill/final_splits`
- Correct rollouts: `/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/distill_results/27b_distill/rollouts`

Output:

- Shard outputs: `/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/distill_results/27b_distill/best_rollouts/shards`
- Split-aligned outputs after merge: `/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/distill_results/27b_distill/best_rollouts/splits`

Run 10 scoring jobs:

```bash
cd /mnt/petrelfs/linjuekai/Multimodel_Reasoning/27b_distill/best-of-n
nohup bash launch.sh best_of_n_config.yaml > logs/qwen35_27b_best_of_n/launch_all.log 2>&1 &
```

Monitor:

```bash
cd /mnt/petrelfs/linjuekai/Multimodel_Reasoning/27b_distill/best-of-n
python3 status.py --config best_of_n_config.yaml
squeue -u "$USER" | grep qwen35_27b_best_of_n
```

Merge after all 10 shards show `stop=True`:

```bash
cd /mnt/petrelfs/linjuekai/Multimodel_Reasoning/27b_distill/best-of-n
python3 merge_shards_to_splits.py --config best_of_n_config.yaml
```

Build one-rollout final JSONL after merge:

```bash
cd /mnt/petrelfs/linjuekai/Multimodel_Reasoning/27b_distill/best-of-n
python3 build_final_rollouts.py --config best_of_n_config.yaml
```

Stop:

```bash
pkill -9 -f 'launch\\.sh .*best_of_n_config\\.yaml' || true
pkill -9 -f 'single_job\\.sh .*best_of_n_config\\.yaml' || true
pkill -9 -f 'slurm_worker\\.sh .*best_of_n_config\\.yaml' || true
squeue -u "$USER" -h -o '%i %j' | awk '$2 ~ /qwen35_27b_best_of_n/ {print $1}' | xargs -r scancel
```
