# Compass Verify for 9B Distill

This project verifies each of the four 9B distill rollouts with OpenCompass CompassVerifier.
Before calling CompassVerifier, each rollout response is checked for the required output format:
`<think>...</think>`, `<answer>...</answer>`, and the last non-empty line must match
`Therefore, the final answer is <answer>...</answer>`. Format failures are directly marked as wrong
and are not sent to CompassVerifier.

Input is read only from:

`/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/qwen35_9b_distill/rollouts`

Outputs are written to:

`/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/qwen35_9b_distill/verify`

Run:

```bash
cd /mnt/petrelfs/linjuekai/Multimodel_Reasoning/compass_verify_9b
nohup bash scripts/launch.sh config/qwen35_9b_compass_verify.yaml > logs/launch_all.log 2>&1 &
```

Monitor:

```bash
python scripts/status.py --config config/qwen35_9b_compass_verify.yaml | tail -1
squeue -u "$USER" | grep qwen35_9b_compass_verify
```

Stop launchers and Slurm jobs:

```bash
pkill -f 'compass_verify_9b/scripts/single_job.sh' || true
squeue -u "$USER" -h -o '%i %j' | awk '$2 ~ /qwen35_9b_compass_verify/ {print $1}' | xargs -r scancel
```
