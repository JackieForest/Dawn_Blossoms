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


DEFAULT_SOURCE_SPLITS = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
    "qwen35_122b_distill/splits"
)
DEFAULT_SOURCE_ROLLOUTS = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
    "qwen35_122b_distill/rollouts"
)
DEFAULT_VERIFY_DIR = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
    "qwen35_122b_distill/verify_response"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
    "distill_results/122b_distill"
)


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("extract_122b_correct_rollouts")
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


def load_verify_correct_rollout_ids(
    verify_path: Path,
    *,
    skip_bad_json: bool,
) -> tuple[dict[str, set[int]], int, int, int]:
    correct_by_key: dict[str, set[int]] = {}
    verify_valid_rows = 0
    total_lines = 0
    bad_json_lines = 0
    if not verify_path.exists():
        return correct_by_key, verify_valid_rows, total_lines, bad_json_lines

    with verify_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if skip_bad_json:
                    bad_json_lines += 1
                    print(f"[WARN] skip bad JSON at {verify_path}:{line_no}: {exc}", file=sys.stderr)
                    continue
                raise ValueError(f"bad json {verify_path}:{line_no}: {exc}") from exc
            if not row.get("verify_valid"):
                continue
            verify_valid_rows += 1
            if int(row.get("correct_count", 0) or 0) <= 0:
                continue
            good_ids = {
                int(j["rollout_id"])
                for j in row.get("judgments", [])
                if isinstance(j, dict)
                and j.get("judgment") == "correct"
                and bool(j.get("format_valid"))
                and "rollout_id" in j
            }
            if good_ids:
                correct_by_key[record_key(row)] = good_ids
    return correct_by_key, verify_valid_rows, total_lines, bad_json_lines


def load_rollout_records(rollout_path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not rollout_path.exists():
        return records
    with rollout_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad json {rollout_path}:{line_no}: {exc}") from exc
            records[record_key(row)] = row
    return records


def filter_correct_rollouts(record: dict[str, Any], keep_ids: set[int]) -> dict[str, Any]:
    rollouts = [
        r
        for r in record.get("rollouts", [])
        if isinstance(r, dict) and int(r.get("rollout_id", -1)) in keep_ids
    ]
    out = dict(record)
    out["rollouts"] = rollouts
    out["num_rollouts"] = len(rollouts)
    out["correct_rollout_ids"] = [int(r.get("rollout_id")) for r in rollouts]
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract 122B samples with correct_count > 0 and keep only correct rollouts."
    )
    parser.add_argument("--source-splits", type=Path, default=DEFAULT_SOURCE_SPLITS)
    parser.add_argument("--source-rollouts", type=Path, default=DEFAULT_SOURCE_ROLLOUTS)
    parser.add_argument("--verify-dir", type=Path, default=DEFAULT_VERIFY_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-splits", type=int, default=100)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--row-group-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-bad-json", action="store_true", default=True)
    parser.add_argument("--no-skip-bad-json", dest="skip_bad_json", action="store_false")
    parser.add_argument("--log-path", type=Path, default=None)
    args = parser.parse_args()

    output_splits = args.output_root / "splits"
    output_rollouts = args.output_root / "rollouts"
    output_splits.mkdir(parents=True, exist_ok=True)
    output_rollouts.mkdir(parents=True, exist_ok=True)

    log_path = args.log_path or (args.output_root / "extract_122b_correct_rollouts.log")
    logger = setup_logger(log_path)
    logger.info("start extract")
    logger.info("source_splits=%s", args.source_splits)
    logger.info("source_rollouts=%s", args.source_rollouts)
    logger.info("verify_dir=%s", args.verify_dir)
    logger.info("output_splits=%s", output_splits)
    logger.info("output_rollouts=%s", output_rollouts)
    logger.info(
        "num_splits=%s compression=%s row_group_size=%s overwrite=%s skip_bad_json=%s",
        args.num_splits,
        args.compression,
        args.row_group_size,
        args.overwrite,
        args.skip_bad_json,
    )

    total_source_rows = 0
    total_verify_valid = 0
    total_verify_lines = 0
    total_bad_json_lines = 0
    total_selected_samples = 0
    total_rollout_records = 0
    total_correct_rollouts = 0
    correct_count_hist: dict[int, int] = {}

    for split in tqdm(range(args.num_splits), desc="extract 122B correct", dynamic_ncols=True):
        source_path = args.source_splits / f"{split}.parquet"
        verify_path = args.verify_dir / f"{split}.jsonl"
        rollout_path = args.source_rollouts / f"{split}.jsonl"
        split_out_path = output_splits / f"{split}.parquet"
        rollout_out_path = output_rollouts / f"{split}.jsonl"

        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if not verify_path.exists():
            raise FileNotFoundError(verify_path)
        if not rollout_path.exists():
            raise FileNotFoundError(rollout_path)
        if not args.overwrite and (split_out_path.exists() or rollout_out_path.exists()):
            raise FileExistsError(f"output for split {split} exists; pass --overwrite to replace it")

        correct_ids_by_key, verify_valid_rows, verify_lines, bad_json_lines = (
            load_verify_correct_rollout_ids(verify_path, skip_bad_json=args.skip_bad_json)
        )
        rollout_records = load_rollout_records(rollout_path)
        df = pd.read_parquet(source_path)
        source_rows = len(df)
        source_keys = set(df["id"].astype(str))

        missing_in_source = sorted(set(correct_ids_by_key) - source_keys)[:10]
        missing_in_rollouts = sorted(set(correct_ids_by_key) - set(rollout_records))[:10]
        if missing_in_source or missing_in_rollouts:
            raise RuntimeError(
                f"split {split}: missing_in_source={missing_in_source} "
                f"missing_in_rollouts={missing_in_rollouts}"
            )

        mask = df["id"].astype(str).isin(correct_ids_by_key)
        out_df = df.loc[mask].copy()
        rollout_rows: list[dict[str, Any]] = []
        for sample_id in out_df["id"].astype(str).tolist():
            row = filter_correct_rollouts(rollout_records[sample_id], correct_ids_by_key[sample_id])
            if not row["rollouts"]:
                raise RuntimeError(f"split {split}: no correct rollout kept for id={sample_id}")
            rollout_rows.append(row)

        split_ids = out_df["id"].astype(str).tolist()
        rollout_ids = [record_key(row) for row in rollout_rows]
        if split_ids != rollout_ids:
            raise RuntimeError(f"split {split}: split ids and rollout ids are not in the same order")

        tmp_split_path = split_out_path.with_name(f"{split_out_path.name}.tmp")
        if tmp_split_path.exists():
            tmp_split_path.unlink()
        out_df.to_parquet(
            tmp_split_path,
            index=False,
            compression=args.compression,
            row_group_size=max(1, args.row_group_size),
        )
        tmp_split_path.replace(split_out_path)
        write_jsonl(rollout_out_path, rollout_rows)

        correct_rollout_count = sum(len(row["rollouts"]) for row in rollout_rows)
        for row in rollout_rows:
            correct_count = len(row["rollouts"])
            correct_count_hist[correct_count] = correct_count_hist.get(correct_count, 0) + 1

        total_source_rows += source_rows
        total_verify_valid += verify_valid_rows
        total_verify_lines += verify_lines
        total_bad_json_lines += bad_json_lines
        total_selected_samples += len(out_df)
        total_rollout_records += len(rollout_rows)
        total_correct_rollouts += correct_rollout_count

        logger.info(
            "split=%s source_rows=%s verify_valid=%s verify_lines=%s bad_json=%s "
            "selected_samples=%s rollout_records=%s correct_rollouts=%s split_path=%s rollout_path=%s",
            split,
            source_rows,
            verify_valid_rows,
            verify_lines,
            bad_json_lines,
            len(out_df),
            len(rollout_rows),
            correct_rollout_count,
            split_out_path,
            rollout_out_path,
        )

    if total_selected_samples != total_rollout_records:
        raise RuntimeError(
            f"global mismatch: selected_samples={total_selected_samples} "
            f"rollout_records={total_rollout_records}"
        )

    logger.info("done")
    logger.info("total_source_rows=%s", total_source_rows)
    logger.info("total_verify_valid_rows=%s", total_verify_valid)
    logger.info("total_verify_lines=%s", total_verify_lines)
    logger.info("total_bad_json_lines=%s", total_bad_json_lines)
    logger.info("total_selected_samples=%s", total_selected_samples)
    logger.info("total_rollout_records=%s", total_rollout_records)
    logger.info("total_correct_rollouts=%s", total_correct_rollouts)
    for correct_count in sorted(correct_count_hist):
        logger.info("correct_rollout_count=%s samples=%s", correct_count, correct_count_hist[correct_count])
    logger.info("id_alignment=ok")
    logger.info("output_splits=%s", output_splits)
    logger.info("output_rollouts=%s", output_rollouts)


if __name__ == "__main__":
    main()
