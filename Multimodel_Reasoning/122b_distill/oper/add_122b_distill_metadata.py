#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


DEFAULT_INPUT_SPLITS = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
    "distill_results/122b_distill/splits"
)
DEFAULT_INPUT_ROLLOUTS = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
    "distill_results/122b_distill/rollouts"
)
DEFAULT_OUTPUT_SPLITS = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
    "distill_results/122b_distill/final_splits"
)


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("add_122b_distill_metadata")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def record_key(row: dict[str, Any]) -> str:
    return str(row.get("id", row.get("index", "")))


def load_correct_counts(rollout_path: Path) -> tuple[list[str], list[int], int]:
    ids: list[str] = []
    counts: list[int] = []
    total_correct_rollouts = 0
    with rollout_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad json {rollout_path}:{line_no}: {exc}") from exc
            correct_count = len(row.get("rollouts", []) or [])
            if correct_count <= 0 or correct_count > 4:
                raise ValueError(f"{rollout_path}:{line_no} invalid correct rollout count: {correct_count}")
            ids.append(record_key(row))
            counts.append(correct_count)
            total_correct_rollouts += correct_count
    return ids, counts, total_correct_rollouts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add distill_model and difficulty metadata to 122B correct distill splits."
    )
    parser.add_argument("--input-splits", type=Path, default=DEFAULT_INPUT_SPLITS)
    parser.add_argument("--input-rollouts", type=Path, default=DEFAULT_INPUT_ROLLOUTS)
    parser.add_argument("--output-splits", type=Path, default=DEFAULT_OUTPUT_SPLITS)
    parser.add_argument("--num-splits", type=int, default=100)
    parser.add_argument("--distill-model", default="qwen3.5-122b")
    parser.add_argument("--rollouts-per-sample", type=int, default=4)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--row-group-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-path", type=Path, default=None)
    args = parser.parse_args()

    args.output_splits.mkdir(parents=True, exist_ok=True)
    log_path = args.log_path or (args.output_splits.parent / "add_122b_distill_metadata.log")
    logger = setup_logger(log_path)

    logger.info("start add metadata")
    logger.info("input_splits=%s", args.input_splits)
    logger.info("input_rollouts=%s", args.input_rollouts)
    logger.info("output_splits=%s", args.output_splits)
    logger.info(
        "num_splits=%s distill_model=%s rollouts_per_sample=%s compression=%s "
        "row_group_size=%s overwrite=%s",
        args.num_splits,
        args.distill_model,
        args.rollouts_per_sample,
        args.compression,
        args.row_group_size,
        args.overwrite,
    )

    total_rows = 0
    total_correct_rollouts = 0
    difficulty_counts: dict[float, int] = {}
    correct_count_hist: dict[int, int] = {}
    domain_counts: dict[str, int] = {}

    for split in tqdm(range(args.num_splits), desc="add 122B metadata", dynamic_ncols=True):
        split_path = args.input_splits / f"{split}.parquet"
        rollout_path = args.input_rollouts / f"{split}.jsonl"
        output_path = args.output_splits / f"{split}.parquet"

        if not split_path.exists():
            raise FileNotFoundError(split_path)
        if not rollout_path.exists():
            raise FileNotFoundError(rollout_path)
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")

        df = pd.read_parquet(split_path)
        rollout_ids, correct_counts, split_correct_rollouts = load_correct_counts(rollout_path)
        split_ids = df["id"].astype(str).tolist()
        if split_ids != rollout_ids:
            preview = []
            for i, (split_id, rollout_id) in enumerate(zip(split_ids, rollout_ids)):
                if split_id != rollout_id:
                    preview.append((i, split_id, rollout_id))
                    if len(preview) >= 5:
                        break
            if len(split_ids) != len(rollout_ids):
                preview.append(("len", len(split_ids), len(rollout_ids)))
            raise RuntimeError(f"split {split}: split ids and rollout ids are not aligned: {preview}")

        difficulties = [count / args.rollouts_per_sample for count in correct_counts]
        out_df = df.copy()
        out_df["distill_model"] = args.distill_model
        out_df["difficulty"] = difficulties

        tmp_path = output_path.with_name(f"{output_path.name}.tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        out_df.to_parquet(
            tmp_path,
            index=False,
            compression=args.compression,
            row_group_size=max(1, args.row_group_size),
        )
        tmp_path.replace(output_path)

        for count, difficulty in zip(correct_counts, difficulties):
            correct_count_hist[count] = correct_count_hist.get(count, 0) + 1
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        if "domain" in out_df.columns:
            for domain, count in out_df["domain"].value_counts(dropna=False).items():
                domain_counts[str(domain)] = domain_counts.get(str(domain), 0) + int(count)
        total_rows += len(out_df)
        total_correct_rollouts += split_correct_rollouts

        logger.info(
            "split=%s rows=%s correct_rollouts=%s difficulty_0.25=%s difficulty_0.5=%s "
            "difficulty_0.75=%s difficulty_1.0=%s path=%s",
            split,
            len(out_df),
            split_correct_rollouts,
            difficulties.count(0.25),
            difficulties.count(0.5),
            difficulties.count(0.75),
            difficulties.count(1.0),
            output_path,
        )

    logger.info("done")
    logger.info("total_rows=%s", total_rows)
    logger.info("total_correct_rollouts=%s", total_correct_rollouts)
    for correct_count in sorted(correct_count_hist):
        logger.info("correct_rollout_count=%s samples=%s", correct_count, correct_count_hist[correct_count])
    for difficulty in sorted(difficulty_counts):
        logger.info("difficulty=%s count=%s", difficulty, difficulty_counts[difficulty])
    for domain, count in sorted(domain_counts.items(), key=lambda item: item[1], reverse=True):
        logger.info("domain=%s count=%s", domain, count)
    logger.info("id_alignment=ok")
    logger.info("output_splits=%s", args.output_splits)


if __name__ == "__main__":
    main()
