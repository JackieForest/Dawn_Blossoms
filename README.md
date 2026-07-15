# Dawn_Blossoms

Personal backup of my multimodal reasoning work before leaving the project.

This repository keeps the code, configs, launch scripts, and notes needed to understand or reuse the experiments. Large runtime artifacts are intentionally excluded, including logs, model weights, checkpoints, outputs, and Python caches.

## Contents

```text
Dawn_Blossoms/
├── Multimodel_Reasoning/   # Main experiment code
├── LlamaFactory/           # LlamaFactory code snapshot used for training/SFT work
├── verl/                   # verl code snapshot used for RL/OPD work
├── .gitignore
└── README.md
```

## Multimodel_Reasoning

Main working directory for multimodal reasoning experiments:

- `4b_difficulty/`, `9b_distill/`, `27b_distill/`, `122b_distill/`: rollout and distillation pipelines.
- `27b_verify/`, `compass_verify_*`, `compassverify_sft_rollouts/`: verifier pipelines.
- `sft_model_rollout/`: SFT model rollout generation.
- `rl/`: RL / GSPO training entrypoints.
- `eval/`: multimodal benchmark evaluation and verification.
- `opd/`: minimal OPD smoke run with verl.

Most subdirectories have their own `README.md` with the exact run, monitor, and stop commands.

## Notes

This backup does not include external data, model checkpoints, generated rollouts, or evaluation outputs. Many scripts still reference the original cluster paths under:

```text
/mnt/dhwfile/raise/user/linjuekai/
/mnt/petrelfs/linjuekai/
```

To reuse the code in another environment, update the model paths, data paths, output roots, Slurm settings, and verifier paths in the corresponding config files.
