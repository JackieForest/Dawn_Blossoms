#!/usr/bin/env python3
from __future__ import annotations

import os


DATASETS = [
    "SFE",
    "MathVision",
    "MMMU_DEV_VAL",
    "MathVista_MINI",
    "VisuLogic",
    "CharXiv_reasoning_val",
    "CharXiv_descriptive_val",
    "LogicVista",
    "ChartQA_TEST",
    "CountBenchQA",
    "OCRBench",
    "RealWorldQA",
    "MathVerse_MINI",
    "AI2D_TEST",
    "ScienceQA_TEST",
    "MMBench_DEV_EN_V11",
    "MMStar",
    "CV-Bench-2D",
    "CV-Bench-3D",
]


def main() -> None:
    from vlmeval.dataset import build_dataset

    print(f"LMUData={os.environ.get('LMUData', '<unset>')}")
    for name in DATASETS:
        try:
            ds = build_dataset(name)
            print(f"{name}: OK rows={len(ds)}")
        except Exception as exc:
            print(f"{name}: FAIL {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
