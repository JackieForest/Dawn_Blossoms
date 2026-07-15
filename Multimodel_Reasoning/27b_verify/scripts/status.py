#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_config


def count_input(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def count_verified(path: Path, valid_key: str, expected_rollouts: int) -> tuple[int, int, int, int]:
    if not path.exists():
        return 0, 0, 0, 0
    valid = correct = wrong = lines = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            judgments = obj.get("judgments", [])
            if obj.get(valid_key) and isinstance(judgments, list) and len(judgments) == expected_rollouts:
                valid += 1
                correct += int(obj.get("correct_count", 0))
                wrong += int(obj.get("wrong_count", 0))
    return valid, correct, wrong, lines


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show 27B verify split progress.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_dir = Path(cfg["input_rollout_dir"])
    output_dir = Path(cfg["output_dir"])
    success_name = str(cfg.get("success_dir_name", "verify"))
    failed_name = str(cfg.get("failed_dir_name", "failed_verify"))
    valid_key = str(cfg.get("valid_key", "verify_valid"))
    expected_rollouts = int(cfg.get("expected_rollouts", 4))
    num_splits = int(cfg.get("num_splits", 100))
    labels_dir = output_dir / success_name
    failed_dir = output_dir / failed_name
    stop_dir = output_dir / "stop_files"

    total_rows = total_done = total_correct = total_wrong = total_fail = total_lines = 0
    for idx in range(num_splits):
        rows = count_input(input_dir / f"{idx}.jsonl")
        done, correct, wrong, success_lines = count_verified(labels_dir / f"{idx}.jsonl", valid_key, expected_rollouts)
        fail = count_lines(failed_dir / f"{idx}.jsonl")
        stop = (stop_dir / f"verify_{idx}.flag").exists()
        total_rows += rows
        total_done += done
        total_correct += correct
        total_wrong += wrong
        total_fail += fail
        total_lines += success_lines
        pct = done / rows * 100 if rows else 0
        print(
            f"split {idx:02d}: rows={rows} done={done} success_lines={success_lines} "
            f"fail_lines={fail} correct_rollouts={correct} wrong_rollouts={wrong} pct={pct:.2f}% stop={stop}"
        )
    pct_total = total_done / total_rows * 100 if total_rows else 0
    rollout_total = total_correct + total_wrong
    acc = total_correct / rollout_total * 100 if rollout_total else 0
    print(
        f"TOTAL: rows={total_rows} done={total_done} success_lines={total_lines} fail_lines={total_fail} "
        f"correct_rollouts={total_correct} wrong_rollouts={total_wrong} rollout_acc={acc:.2f}% pct={pct_total:.2f}%"
    )


if __name__ == "__main__":
    main()
