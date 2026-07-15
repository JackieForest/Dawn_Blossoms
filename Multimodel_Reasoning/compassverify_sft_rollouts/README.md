# Compass Verify for SFT RL Rollouts

This project verifies each of the eight SFT rollout responses for every RL-pool sample with OpenCompass CompassVerifier.

It is copied from the `compass_verify_122b` workflow, but points only at the SFT rollout outputs below. The input rollout files are read-only for this project.

Input rollouts:

```text
/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/sft_model_rollout/output/qwen35_4b_full_sft_rl_rollout/rollouts
```

Verification outputs:

```text
/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/sft_model_rollout/output/qwen35_4b_full_sft_rl_rollout/verify_response
/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/sft_model_rollout/output/qwen35_4b_full_sft_rl_rollout/failed_verify_response
/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/sft_model_rollout/output/qwen35_4b_full_sft_rl_rollout/verify_response_stop_files
```

Run all 400 splits:

```bash
cd /mnt/petrelfs/linjuekai/Multimodel_Reasoning/compassverify_sft_rollouts
nohup bash scripts/launch.sh config/qwen35_4b_full_sft_rl_rollout_compass_verify.yaml > logs/qwen35_4b_full_sft_rl_rollout_compass_verify/launch_all.log 2>&1 &
```

Monitor:

```bash
python3 scripts/status.py --config config/qwen35_4b_full_sft_rl_rollout_compass_verify.yaml | tail -1
squeue -u "$USER" | grep qwen35_4b_full_sft_rl_rollout_compass_verify
```

Stop launchers and Slurm jobs:

```bash
pkill -f 'compassverify_sft_rollouts/scripts/single_job.sh' || true
squeue -u "$USER" -h -o '%i %j' | awk '$2 ~ /qwen35_4b_full_sft_rl_rollout_compass_verify/ {print $1}' | xargs -r scancel
```

Each output row contains `correct_count`, `wrong_count`, `invalid_count`, `parse_failed_count`, and the per-rollout `judgments` list.
