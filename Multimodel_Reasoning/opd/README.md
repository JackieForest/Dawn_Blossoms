# MMR OPD Smoke Run

This folder contains a minimal OPD launch path for checking that verl can run
student rollout, teacher top-k logprob collection, and student update.

Default paths:

- raw student: `/mnt/dhwfile/raise/user/linjuekai/models/Qwen3.5-4B-Base`
- patched student used by OPD: `/mnt/petrelfs/linjuekai/Multimodel_Reasoning/opd/models/Qwen3.5-4B-Base-chat-template`
- teacher: `/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/training/models/qwen35_4b_base_full_distill_sft_528k_16k`
- data: `/mnt/petrelfs/linjuekai/Multimodel_Reasoning/rl_data/new_rl_data/train_*.parquet`

The patched student directory is a lightweight symlink directory that adds
`chat_template.jinja` from the teacher/SFT model so verl's multimodal processor
can format prompts.

Run a tiny smoke test:

```bash
cd /mnt/petrelfs/linjuekai/Multimodel_Reasoning/opd
bash run_opd_smoke.sh
```

Submit the same smoke test to Slurm:

```bash
cd /mnt/petrelfs/linjuekai/Multimodel_Reasoning/opd
sbatch submit_opd_smoke.sh
```

The Slurm smoke defaults request 4 GPUs. Inside verl this is split as
`trainer.n_gpus_per_node=2` for the student/main pool plus
`TEACHER_NGPUS=2`, `TEACHER_TP=2` for the distillation teacher pool.

Useful overrides:

```bash
SMOKE_ROWS_PER_DOMAIN=64 TRAIN_STEPS=10 bash run_opd_smoke.sh
USE_SMOKE_DATA=0 TRAIN_STEPS=100 bash run_opd_smoke.sh
sbatch --export=ALL,SMOKE_ROWS_PER_DOMAIN=64,TRAIN_STEPS=10,SAVE_FREQ=5 submit_opd_smoke.sh
```

Inspect routing fields:

```bash
python3 inspect_opd_data.py
```

Current note: the five RL parquet files all have `data_source=scimm`, so this
smoke script uses a single teacher. For 5-domain MOPD, add a flat routing field
or regenerate `data_source` values per domain before adding five teachers.
