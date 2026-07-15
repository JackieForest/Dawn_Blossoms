#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_VERIFY_DIR = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
    "qwen35_9b_distill/verify_response"
)
DEFAULT_INPUT_DIR = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
    "qwen35_9b_distill/rollouts"
)


def record_key(row: dict[str, Any]) -> str:
    return str(row.get("id", row.get("index", "")))


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_latest_valid(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] skip bad json {path}:{line_no}: {exc}")
                continue
            if row.get("verify_valid"):
                latest[record_key(row)] = row
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count how many 9B distill samples are correct in 0/4...4/4 rollouts."
    )
    parser.add_argument("--verify-dir", type=Path, default=DEFAULT_VERIFY_DIR)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--num-splits", type=int, default=400)
    parser.add_argument("--expected-rollouts", type=int, default=4)
    args = parser.parse_args()

    dist: Counter[int] = Counter()
    total_input = 0
    total_valid = 0
    missing_splits: list[int] = []
    incomplete_splits: list[tuple[int, int, int]] = []

    for split in range(args.num_splits):
        input_rows = count_lines(args.input_dir / f"{split}.jsonl")
        latest = load_latest_valid(args.verify_dir / f"{split}.jsonl")
        valid_rows = len(latest)
        total_input += input_rows
        total_valid += valid_rows

        if input_rows and valid_rows < input_rows:
            incomplete_splits.append((split, valid_rows, input_rows))
        if input_rows and not latest:
            missing_splits.append(split)

        for row in latest.values():
            correct = int(row.get("correct_count", 0) or 0)
            if correct < 0 or correct > args.expected_rollouts:
                print(f"[WARN] unusual correct_count={correct} key={record_key(row)} split={split}")
            dist[correct] += 1

    print(f"verify_dir: {args.verify_dir}")
    print(f"input_dir:  {args.input_dir}")
    print(f"splits: {args.num_splits}")
    print(f"rows: valid={total_valid} input={total_input} missing={total_input - total_valid}")
    print()
    print("correct_count distribution:")
    for correct in range(args.expected_rollouts + 1):
        count = dist[correct]
        pct = count / total_valid * 100 if total_valid else 0.0
        print(f"  {correct}/{args.expected_rollouts}: {count} ({pct:.2f}%)")
    print()
    at_least_one = sum(count for correct, count in dist.items() if correct >= 1)
    all_wrong = dist[0]
    print(f"at_least_1_correct: {at_least_one} ({at_least_one / total_valid * 100 if total_valid else 0:.2f}%)")
    print(f"all_wrong_0_correct: {all_wrong} ({all_wrong / total_valid * 100 if total_valid else 0:.2f}%)")

    if incomplete_splits:
        preview = ", ".join(f"{idx}:{done}/{total}" for idx, done, total in incomplete_splits[:20])
        suffix = " ..." if len(incomplete_splits) > 20 else ""
        print(f"[WARN] incomplete_splits={len(incomplete_splits)} {preview}{suffix}")
    if missing_splits:
        preview = ", ".join(map(str, missing_splits[:20]))
        suffix = " ..." if len(missing_splits) > 20 else ""
        print(f"[WARN] missing_splits={len(missing_splits)} {preview}{suffix}")


if __name__ == "__main__":
    main()
