#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    args = parser.parse_args()

    output_root = Path(os.environ.get("EVAL_OUTPUT_ROOT", str(Path(__file__).resolve().parent)))
    out_dir = output_root / "outputs" / args.model_name
    rollout = out_dir / "rollout.jsonl"
    if not rollout.exists():
        print(f"No rollout file: {rollout}")
        return

    counts = Counter()
    with rollout.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ds = obj.get("dataset", "unknown")
            if ds == "MMMU_DEV_VAL":
                split = str(obj.get("split", "")).lower()
                ds = "MMMU_DEV" if split == "dev" else "MMMU_VAL"
            counts[ds] += 1

    print(f"rollout rows: {sum(counts.values())}")
    for ds, n in sorted(counts.items()):
        acc = out_dir / f"{ds}_acc.json"
        res = out_dir / f"{ds}_results.jsonl"
        suffix = ""
        if acc.exists():
            try:
                data = json.loads(acc.read_text())
                suffix = f" verified={data.get('total')} acc={float(data.get('accuracy', 0))*100:.2f}%"
            except Exception:
                suffix = " verified=?"
        elif res.exists():
            suffix = " results_exist_no_acc"
        print(f"{ds}: rollout={n}{suffix}")


if __name__ == "__main__":
    main()
