#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split qwen35 122b left parquet into even shards.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
            "qwen35_122b_distill_left/qwen35_122b_distill_left.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
            "qwen35_122b_distill_left/splits"
        ),
    )
    parser.add_argument("--num-splits", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_sizes(total: int, num_splits: int) -> list[int]:
    base = total // num_splits
    rem = total % num_splits
    return [base + (1 if i < rem else 0) for i in range(num_splits)]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.num_splits <= 0:
        raise ValueError("--num-splits must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.output_dir.glob("*.parquet"))
    if existing and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} already has parquet files; pass --overwrite")

    for path in existing:
        path.unlink()

    table = pq.read_table(args.input)
    sizes = split_sizes(table.num_rows, args.num_splits)
    logging.info("input=%s rows=%d output_dir=%s", args.input, table.num_rows, args.output_dir)
    logging.info("num_splits=%d min_size=%d max_size=%d", args.num_splits, min(sizes), max(sizes))

    offset = 0
    for idx, size in enumerate(sizes):
        shard = table.slice(offset, size)
        output = args.output_dir / f"{idx}.parquet"
        tmp_output = output.with_suffix(".parquet.tmp")
        if tmp_output.exists():
            tmp_output.unlink()
        pq.write_table(shard, tmp_output, compression="snappy")
        os.replace(tmp_output, output)
        logging.info("wrote split=%02d rows=%d path=%s", idx, size, output)
        offset += size

    logging.info("done rows=%d written_rows=%d", table.num_rows, offset)


if __name__ == "__main__":
    main()
