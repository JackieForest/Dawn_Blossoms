# Multimodal Evaluation

This directory runs the default 19-benchmark evaluation and verifies predictions with CompassVerifier.

Default benchmarks:

```text
SFE, MathVision, MMMU_DEV_VAL, MathVista_MINI, VisuLogic,
CharXiv_reasoning_val, CharXiv_descriptive_val, LogicVista,
ChartQA_TEST, OCRBench, RealWorldQA, MathVerse_MINI,
AI2D_TEST, ScienceQA_TEST, MMBench_DEV_EN_V11, MMStar,
CV-Bench-2D, CV-Bench-3D
```

`CountBenchQA` is also integrated and can be run by setting `DATASET=CountBenchQA`.

Run:

```bash
cd /mnt/petrelfs/linjuekai/Multimodel_Reasoning/eval
nohup bash run.sh /path/to/model 8 > launcher_logs/model.log 2>&1 &
```

Use the SFT training system prompt only for finetuned training models:

```bash
EVAL_SYSTEM_PROMPT_MODE=sft nohup bash run.sh /path/to/model 8 > launcher_logs/model.log 2>&1 &
```

Only verify an existing rollout:

```bash
bash verify.sh outputs/<model_name>
```

By default, outputs and caches are written under:

```text
/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/eval
```

Key paths:

```text
$EVAL_OUTPUT_ROOT/outputs/<model_name>/
$EVAL_OUTPUT_ROOT/cache/
```
