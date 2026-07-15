#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any


TASK_MAPPING = {
    "SFE": "SFE_acc.json",
    "CV-Bench-2D": "CV-Bench-2D_acc.json",
    "CV-Bench-3D": "CV-Bench-3D_acc.json",
    "MathVista_MINI": "MathVista_MINI_acc.json",
    "MathVision": "MathVision_acc.json",
    "MathVerse_MINI": "MathVerse_MINI_acc.json",
    "LogicVista": "LogicVista_acc.json",
    "VisuLogic": "VisuLogic_acc.json",
    "AI2D_TEST": "AI2D_TEST_acc.json",
    "ScienceQA_TEST": "ScienceQA_TEST_acc.json",
    "MMBench_DEV_EN_V11": "MMBench_DEV_EN_V11_acc.json",
    "RealWorldQA": "RealWorldQA_acc.json",
    "MMStar": "MMStar_acc.json",
    "MMMU_VAL": "MMMU_VAL_acc.json",
    "MMMU_DEV": "MMMU_DEV_acc.json",
    "ChartQA_TEST": "ChartQA_TEST_acc.json",
    "CountBenchQA": "CountBenchQA_acc.json",
    "OCRBench": "OCRBench_acc.json",
    "CharXiv_reasoning_val": "CharXiv_reasoning_val_acc.json",
    "CharXiv_descriptive_val": "CharXiv_descriptive_val_acc.json",
}

REPORT_ORDER = [
    ("SFE", "AVG_SCI"),
    ("AVG_SCI", "AVG_SCI_CALC"),
    ("CV-Bench-2D", "AVG_CV"),
    ("CV-Bench-3D", "AVG_CV"),
    ("AVG_CV", "AVG_CV_CALC"),
    ("MathVista_MINI", "AVG_REASON"),
    ("MathVision", "AVG_REASON"),
    ("MathVerse_MINI", "AVG_REASON"),
    ("LogicVista", "AVG_REASON"),
    ("VisuLogic", "AVG_REASON"),
    ("AI2D_TEST", "AVG_REASON"),
    ("ScienceQA_TEST", "AVG_REASON"),
    ("AVG_REASON", "AVG_REASON_CALC"),
    ("MMBench_DEV_EN_V11", "AVG_GENERAL"),
    ("RealWorldQA", "AVG_GENERAL"),
    ("MMStar", "AVG_GENERAL"),
    ("MMMU_VAL", "AVG_GENERAL"),
    ("MMMU_DEV", "AVG_GENERAL"),
    ("AVG_GENERAL", "AVG_GENERAL_CALC"),
    ("ChartQA_TEST", "AVG_CHART_OCR"),
    ("CountBenchQA", "AVG_CHART_OCR"),
    ("OCRBench", "AVG_CHART_OCR"),
    ("CharXiv_reasoning_val", "AVG_CHART_OCR"),
    ("CharXiv_descriptive_val", "AVG_CHART_OCR"),
    ("AVG_CHART_OCR", "AVG_CHART_OCR_CALC"),
    ("AVG_ALL", "AVG_ALL_CALC"),
]


def load_accuracy(base_path: str, filename: str | None) -> float | str:
    if not filename:
        return "NA"
    path = os.path.join(base_path, filename)
    if not os.path.exists(path):
        return "NA"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data.get("accuracy", "nan")) * 100


def average(values: list[Any]) -> float | str:
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return "NA"
    return sum(nums) / len(nums)


def save_single_row_csv(base_path: str) -> None:
    all_results = {}
    grouped_values = {
        "AVG_CV": [],
        "AVG_SCI": [],
        "AVG_REASON": [],
        "AVG_GENERAL": [],
        "AVG_CHART_OCR": [],
        "AVG_ALL": [],
    }

    for task_name, group_key in REPORT_ORDER:
        if task_name.startswith("AVG"):
            continue
        val = load_accuracy(base_path, TASK_MAPPING.get(task_name))
        all_results[task_name] = val
        if group_key in grouped_values:
            grouped_values[group_key].append(val)
            grouped_values["AVG_ALL"].append(val)

    avg_results = {
        "AVG_CV_CALC": average(grouped_values["AVG_CV"]),
        "AVG_SCI_CALC": average(grouped_values["AVG_SCI"]),
        "AVG_REASON_CALC": average(grouped_values["AVG_REASON"]),
        "AVG_GENERAL_CALC": average(grouped_values["AVG_GENERAL"]),
        "AVG_CHART_OCR_CALC": average(grouped_values["AVG_CHART_OCR"]),
        "AVG_ALL_CALC": average(grouped_values["AVG_ALL"]),
    }

    headers = []
    values = []
    for task_name, group_key in REPORT_ORDER:
        headers.append(task_name)
        val = avg_results[group_key] if task_name.startswith("AVG") else all_results[task_name]
        values.append(f"{val:.4f}" if isinstance(val, float) else str(val))

    out_file = os.path.join(base_path, "accuracy_row.csv")
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(values)
    print(f"Saved {out_file}")
    print(", ".join(headers))
    print(", ".join(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()
    output_root = os.environ.get("EVAL_OUTPUT_ROOT", os.path.dirname(__file__))
    base_path = os.path.join(output_root, "outputs", args.model_name)
    save_single_row_csv(base_path)


if __name__ == "__main__":
    main()
