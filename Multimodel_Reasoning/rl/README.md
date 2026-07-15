# MMR RL GSPO + CompassVerifier

This directory contains the RL training entrypoints for the selected domain data.

## Data

The converted verl-format parquet files are:

- `train_science_compass.parquet`
- `train_chart_table_doc_compass.parquet`
- `train_math_compass.parquet`
- `train_logic_game_puzzle_compass.parquet`
- `train_spatial_general_compass.parquet`

Each row has `prompt`, `images`, `answer`, `reward_model`, `data_source`, and `extra_info`.

## Reward

`compass_format_reward.py` computes:

```text
reward = 0.9 * correctness_reward + 0.1 * format_reward
```

Correctness is judged by CompassVerifier-7B with the same `CV_COT_PROMPT` as
`/mnt/petrelfs/linjuekai/Multimodel_Reasoning/eval/verify.py`.

## Run One Domain

```bash
cd /mnt/petrelfs/linjuekai/Multimodel_Reasoning/rl
sbatch submit_gspo_compass_domain.sh science
```

Valid domains:

- `science`
- `chart_table_doc`
- `math`
- `logic_game_puzzle`
- `spatial_general`

## Run All Domains

```bash
cd /mnt/petrelfs/linjuekai/Multimodel_Reasoning/rl
bash submit_all_domains.sh
```

## Useful Overrides

```bash
MODEL_PATH=/path/to/model \
TRAIN_BATCH_SIZE=256 \
TRAIN_STEPS=300 \
MAX_RESPONSE_LENGTH=16384 \
sbatch submit_gspo_compass_domain.sh math
```

Each job requests 12 GPUs. GPUs `0-7` are used for RL training and GPUs `8-11`
serve CompassVerifier.

## Default Training Config

- Optimizer: `AdamW`
- Learning rate: `1e-6`
- Scheduler: `constant`
- Weight decay: `0.1`
- Train steps: `300`
- Warmup steps: `10`
- Batch size: `128`
- Prompt length: `4096`
- Output length: `16384`
- Temperature: `1.0`
- Rollouts per prompt: `8`
- GSPO clip: `clip_ratio_low=3e-4`, `clip_ratio_high=4e-4`
- CompassVerifier max output tokens: `2048`
- CompassVerifier overlong input handling: if the verifier prompt is longer
  than `30000` tokens, keep the last `30000` candidate-response tokens and
  shrink further only if the question/answer/prompt overhead still exceeds the
  limit.
- Save frequency: every `10` steps
- Checkpoint root: `/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/training/rl`
