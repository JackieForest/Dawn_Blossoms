# 122B Answer Likelihood Best-of-N

Select one best correct rollout for each 122B-correct sample.

Run:

```bash
cd /mnt/petrelfs/linjuekai/Multimodel_Reasoning/122b_distill/best-of-n
mkdir -p logs/qwen35_122b_best_of_n
nohup bash launch.sh best_of_n_122b_config.yaml > logs/qwen35_122b_best_of_n/launch_all.log 2>&1 &
```

Monitor:

```bash
python3 status.py --config best_of_n_122b_config.yaml
squeue -u "$USER" | grep qwen35_122b_best_of_n
```

After all shards stop:

```bash
python3 merge_shards_to_splits.py --config best_of_n_122b_config.yaml
python3 build_final_rollouts.py \
  --config best_of_n_122b_config.yaml \
  --output-dir /mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/distill_results/122b_distill/final_rollouts \
  --overwrite
```
