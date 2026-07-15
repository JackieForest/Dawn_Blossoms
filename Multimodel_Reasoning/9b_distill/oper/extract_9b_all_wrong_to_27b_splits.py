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
    "qwen35_9b_distill/splits"
)
DEFAULT_VERIFY_DIR = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
    "qwen35_9b_distill/verify_response"
)
DEFAULT_OUTPUT_SPLITS = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
    "qwen35_27b_distill/splits"
)


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("extract_9b_all_wrong")
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


def load_all_wrong_keys(verify_path: Path) -> tuple[set[str], int, int]:
    latest_valid: dict[str, dict[str, Any]] = {}
    total_lines = 0
    if not verify_path.exists():
        return set(), 0, 0

    with verify_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad json {verify_path}:{line_no}: {exc}") from exc
            if row.get("verify_valid"):
                latest_valid[record_key(row)] = row

    all_wrong = {
        key
        for key, row in latest_valid.items()
        if int(row.get("correct_count", 0) or 0) == 0
    }
    return all_wrong, len(latest_valid), total_lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract samples whose 9B four rollouts are all wrong into 27B distill splits."
    )
    parser.add_argument("--source-splits", type=Path, default=DEFAULT_SOURCE_SPLITS)
    parser.add_argument("--verify-dir", type=Path, default=DEFAULT_VERIFY_DIR)
    parser.add_argument("--output-splits", type=Path, default=DEFAULT_OUTPUT_SPLITS)
    parser.add_argument("--num-splits", type=int, default=400)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip output splits that already exist.")
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=128,
        help="Rows per parquet row group. Small groups avoid oversized compressed pages for image-heavy rows.",
    )
    parser.add_argument("--log-path", type=Path, default=None)
    args = parser.parse_args()

    log_path = args.log_path or (args.output_splits / "extract_9b_all_wrong_to_27b_splits.log")
    logger = setup_logger(log_path)

    logger.info("start extract")
    logger.info("source_splits=%s", args.source_splits)
    logger.info("verify_dir=%s", args.verify_dir)
    logger.info("output_splits=%s", args.output_splits)
    logger.info(
        "num_splits=%s compression=%s overwrite=%s resume=%s row_group_size=%s",
        args.num_splits,
        args.compression,
        args.overwrite,
        args.resume,
        args.row_group_size,
    )

    if not args.source_splits.exists():
        raise FileNotFoundError(args.source_splits)
    if not args.verify_dir.exists():
        raise FileNotFoundError(args.verify_dir)
    args.output_splits.mkdir(parents=True, exist_ok=True)

    total_source = 0
    total_valid_verify = 0
    total_all_wrong = 0
    total_written = 0
    incomplete: list[tuple[int, int, int]] = []

    for split in tqdm(range(args.num_splits), desc="extract splits", dynamic_ncols=True):
        source_path = args.source_splits / f"{split}.parquet"
        verify_path = args.verify_dir / f"{split}.jsonl"
        output_path = args.output_splits / f"{split}.parquet"

        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if output_path.exists() and args.resume and not args.overwrite:
            logger.info("split=%s skip_existing path=%s", split, output_path)
            continue
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")

        all_wrong_keys, valid_verify_rows, verify_lines = load_all_wrong_keys(verify_path)
        df = pd.read_parquet(source_path)
        source_rows = len(df)

        total_source += source_rows
        total_valid_verify += valid_verify_rows
        total_all_wrong += len(all_wrong_keys)
        if valid_verify_rows != source_rows:
            incomplete.append((split, valid_verify_rows, source_rows))

        mask = df["id"].astype(str).isin(all_wrong_keys)
        out_df = df.loc[mask].copy()
        written = len(out_df)
        total_written += written
        if written != len(all_wrong_keys):
            source_keys = set(df["id"].astype(str))
            missing = sorted(all_wrong_keys - source_keys)[:10]
            raise RuntimeError(
                f"split {split}: all_wrong_keys={len(all_wrong_keys)} written={written} "
                f"missing_in_source_preview={missing}"
            )

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
        logger.info(
            "split=%s source_rows=%s verify_valid=%s verify_lines=%s all_wrong=%s wrote=%s path=%s",
            split,
            source_rows,
            valid_verify_rows,
            verify_lines,
            len(all_wrong_keys),
            written,
            output_path,
        )

    logger.info("done")
    logger.info("total_source_rows=%s", total_source)
    logger.info("total_valid_verify_rows=%s", total_valid_verify)
    logger.info("total_all_wrong=%s", total_all_wrong)
    logger.info("total_written=%s", total_written)
    logger.info("output_splits=%s", args.output_splits)
    if incomplete:
        preview = ", ".join(f"{idx}:{valid}/{src}" for idx, valid, src in incomplete[:20])
        suffix = " ..." if len(incomplete) > 20 else ""
        logger.warning("incomplete_verify_splits=%s %s%s", len(incomplete), preview, suffix)


if __name__ == "__main__":
    main()
