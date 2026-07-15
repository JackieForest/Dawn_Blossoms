#!/usr/bin/env python3
"""Summarize 27B judge results for 4-rollout verification."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_VERIFY_DIR = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/"
    "rollouts/output/qwen35_27b_verify/verify"
)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad json in {path}:{line_no}: {exc}") from exc


def new_bucket(expected_rollouts: int) -> dict:
    return {
        "samples": 0,
        "correct_rollouts": 0,
        "total_rollouts": 0,
        "distribution": Counter({i: 0 for i in range(expected_rollouts + 1)}),
    }


def update_bucket(bucket: dict, correct_count: int, num_rollouts: int) -> None:
    bucket["samples"] += 1
    bucket["correct_rollouts"] += correct_count
    bucket["total_rollouts"] += num_rollouts
    bucket["distribution"][correct_count] += 1


def format_bucket(name: str, bucket: dict, expected_rollouts: int) -> str:
    samples = bucket["samples"]
    total_rollouts = bucket["total_rollouts"]
    correct_rollouts = bucket["correct_rollouts"]
    rollout_acc = correct_rollouts / total_rollouts if total_rollouts else 0.0
    parts = [
        f"{i}/{expected_rollouts}={bucket['distribution'][i]} "
        f"({bucket['distribution'][i] / samples * 100:.2f}%)"
        for i in range(expected_rollouts + 1)
    ]
    return (
        f"{name}: samples={samples}, rollout_acc={rollout_acc * 100:.2f}%, "
        + ", ".join(parts)
    )


def serialize_bucket(bucket: dict, expected_rollouts: int) -> dict:
    samples = bucket["samples"]
    total_rollouts = bucket["total_rollouts"]
    correct_rollouts = bucket["correct_rollouts"]
    return {
        "samples": samples,
        "total_rollouts": total_rollouts,
        "correct_rollouts": correct_rollouts,
        "rollout_accuracy": correct_rollouts / total_rollouts if total_rollouts else 0.0,
        "distribution": {
            str(i): {
                "samples": bucket["distribution"][i],
                "rate": bucket["distribution"][i] / samples if samples else 0.0,
            }
            for i in range(expected_rollouts + 1)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count distribution of correct rollouts per sample."
    )
    parser.add_argument(
        "--verify-dir",
        type=Path,
        default=DEFAULT_VERIFY_DIR,
        help=f"Directory containing verified split jsonl files. Default: {DEFAULT_VERIFY_DIR}",
    )
    parser.add_argument(
        "--expected-rollouts",
        type=int,
        default=4,
        help="Expected number of rollouts per sample.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the summary JSON.",
    )
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=["domain", "type", "subtype"],
        help="Metadata fields to group by. Default: domain type subtype.",
    )
    args = parser.parse_args()

    files = sorted(args.verify_dir.glob("*.jsonl"), key=lambda p: int(p.stem))
    if not files:
        raise SystemExit(f"no jsonl files found in {args.verify_dir}")

    dist: Counter[int] = Counter()
    invalid = 0
    total_samples = 0
    total_correct_rollouts = 0
    total_rollouts = 0
    grouped = {
        field: defaultdict(lambda: new_bucket(args.expected_rollouts))
        for field in args.group_by
    }

    for path in files:
        for obj in iter_jsonl(path):
            total_samples += 1
            if not obj.get("verify_valid"):
                invalid += 1
                continue
            num_rollouts = int(obj.get("num_rollouts", args.expected_rollouts))
            correct_count = int(obj.get("correct_count", 0))
            if num_rollouts != args.expected_rollouts:
                invalid += 1
                continue
            dist[correct_count] += 1
            total_correct_rollouts += correct_count
            total_rollouts += num_rollouts
            for field, buckets in grouped.items():
                value = obj.get(field)
                key = str(value) if value not in (None, "") else "<missing>"
                update_bucket(buckets[key], correct_count, num_rollouts)

    valid_samples = sum(dist.values())
    rollout_acc = total_correct_rollouts / total_rollouts if total_rollouts else 0.0
    at_least_one = sum(count for correct, count in dist.items() if correct >= 1)
    all_correct = dist[args.expected_rollouts]
    majority_correct = sum(
        count for correct, count in dist.items() if correct > args.expected_rollouts / 2
    )

    summary = {
        "verify_dir": str(args.verify_dir),
        "num_files": len(files),
        "expected_rollouts": args.expected_rollouts,
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "invalid_samples": invalid,
        "total_rollouts": total_rollouts,
        "total_correct_rollouts": total_correct_rollouts,
        "rollout_accuracy": rollout_acc,
        "at_least_one_correct_samples": at_least_one,
        "at_least_one_correct_rate": at_least_one / valid_samples if valid_samples else 0.0,
        "majority_correct_samples": majority_correct,
        "majority_correct_rate": majority_correct / valid_samples if valid_samples else 0.0,
        "all_correct_samples": all_correct,
        "all_correct_rate": all_correct / valid_samples if valid_samples else 0.0,
        "distribution": {
            str(i): {
                "samples": dist[i],
                "rate": dist[i] / valid_samples if valid_samples else 0.0,
            }
            for i in range(args.expected_rollouts + 1)
        },
        "by_category": {
            field: {
                key: serialize_bucket(bucket, args.expected_rollouts)
                for key, bucket in sorted(
                    buckets.items(), key=lambda item: (-item[1]["samples"], item[0])
                )
            }
            for field, buckets in grouped.items()
        },
    }

    print(f"verify_dir: {args.verify_dir}")
    print(f"files: {len(files)}")
    print(f"samples: valid={valid_samples} invalid={invalid} total={total_samples}")
    print(
        "rollout_acc: "
        f"{total_correct_rollouts}/{total_rollouts} = {rollout_acc * 100:.2f}%"
    )
    print("\ncorrect_count distribution:")
    for i in range(args.expected_rollouts + 1):
        count = dist[i]
        rate = count / valid_samples if valid_samples else 0.0
        print(f"  {i}/{args.expected_rollouts}: {count} ({rate * 100:.2f}%)")
    print(
        "\nsummary: "
        f">=1 correct {at_least_one} ({summary['at_least_one_correct_rate'] * 100:.2f}%), "
        f"majority correct {majority_correct} ({summary['majority_correct_rate'] * 100:.2f}%), "
        f"all correct {all_correct} ({summary['all_correct_rate'] * 100:.2f}%)"
    )
    for field, buckets in grouped.items():
        print(f"\nby {field}:")
        for key, bucket in sorted(
            buckets.items(), key=lambda item: (-item[1]["samples"], item[0])
        ):
            print("  " + format_bucket(key, bucket, args.expected_rollouts))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote summary: {args.output_json}")


if __name__ == "__main__":
    main()
