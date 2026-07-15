# Qwen3.5-4B Thinking Rollouts

This folder runs four independent thinking rollouts per sample with Qwen3.5-4B.

Input data:

```text
/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/qwen35_4b_pr/splits/Pool.parquet
```

Numeric split files are written under the same `splits` directory as `0.parquet` ... `99.parquet`.

## 1. Split Pool.parquet into 100 shards

```bash
cd /mnt/petrelfs/linjuekai/Multimodel_Reasoning/4b_difficulty
python scripts/split.py --config config/qwen35_4b_pr.yaml --resume
```

Use `--overwrite` only when you intentionally want to rebuild numeric split files.

## 2. Launch all 100 jobs

```bash
bash scripts/launch.sh config/qwen35_4b_pr.yaml
```

Each split job requests one GPU, starts one vLLM server, and sends four separate requests per sample.

## 3. Check progress

```bash
python scripts/status.py --config config/qwen35_4b_pr.yaml
squeue -u "$USER" -h -o '%i %j %T %R' | grep qwen35_4b_pr || true
```

## 4. Stop jobs

```bash
cd /mnt/petrelfs/linjuekai/Multimodel_Reasoning/4b_difficulty
pkill -9 -f 'launch\\.sh config/qwen35_4b_pr\\.yaml' || true
pkill -9 -f 'single_job\\.sh config/qwen35_4b_pr\\.yaml' || true
pkill -9 -f 'slurm_worker\\.sh config/qwen35_4b_pr\\.yaml' || true
squeue -u "$USER" -h -o '%i %j' | awk '$2 ~ /qwen35_4b_pr/ {print $1}' | xargs -r scancel
```

Outputs:

```text
/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/qwen35_4b_pr/rollouts
/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/qwen35_4b_pr/failed_rollouts
```
