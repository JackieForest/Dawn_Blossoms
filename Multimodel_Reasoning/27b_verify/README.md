# Qwen3.5-27B Verify

This project verifies Qwen3.5-4B rollouts with Qwen3.5-27B as a judge.

Run from this directory:

```bash
nohup bash scripts/launch.sh config/qwen35_27b_verify.yaml > logs/qwen35_27b_verify/launch_all.log 2>&1 &
```

Check progress:

```bash
python3 scripts/status.py --config config/qwen35_27b_verify.yaml
```
