#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a parquet containing original rows that do not have valid rollouts."
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path(
            "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
            "qwen35_122b_distill/splits"
        ),
    )
    parser.add_argument(
        "--rollouts-dir",
        type=Path,
        default=Path(
            "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
            "qwen35_122b_distill/rollouts"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
            "qwen35_122b_distill_left/qwen35_122b_distill_left.parquet"
        ),
    )
    parser.add_argument("--valid-key", default="rollout_valid")
    parser.add_argument("--expected-rollouts", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_done_ids(rollouts_dir: Path, valid_key: str, expected_rollouts: int) -> set[str]:
    done_ids: set[str] = set()
    paths = sorted(
        [p for p in rollouts_dir.glob("*.jsonl") if p.stem.isdigit()],
        key=lambda p: int(p.stem),
    )
    for path in paths:
        before = len(done_ids)
        bad_json = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad_json += 1
                    continue
                row_id = obj.get("id")
                rollouts = obj.get("rollouts")
                if (
                    isinstance(row_id, str)
                    and obj.get(valid_key)
                    and isinstance(rollouts, list)
                    and len(rollouts) == expected_rollouts
                ):
                    done_ids.add(row_id)
        logging.info(
            "loaded done ids from %s: +%d total=%d bad_json=%d",
            path.name,
            len(done_ids) - before,
            len(done_ids),
            bad_json,
        )
    return done_ids


def build_left_table(splits_dir: Path, done_ids: set[str], output: Path) -> tuple[int, int]:
    split_paths = sorted(
        [p for p in splits_dir.glob("*.parquet") if p.stem.isdigit()],
        key=lambda p: int(p.stem),
    )
    if not split_paths:
        raise FileNotFoundError(f"No numeric parquet splits found under {splits_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    done_ids_array = pa.array(sorted(done_ids), type=pa.string())
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    left_rows = 0

    try:
        for path in split_paths:
            table = pq.read_table(path)
            if writer is None:
                writer = pq.ParquetWriter(tmp_output, table.schema, compression="snappy")

            ids = table["id"]
            is_done = pc.is_in(ids, value_set=done_ids_array)
            keep = pc.invert(is_done)
            left_table = table.filter(keep)

            rows = table.num_rows
            left = left_table.num_rows
            total_rows += rows
            left_rows += left
            if left:
                writer.write_table(left_table)

            logging.info(
                "split=%s rows=%d done=%d left=%d total_left=%d",
                path.stem,
                rows,
                rows - left,
                left,
                left_rows,
            )
    finally:
        if writer is not None:
            writer.close()

    os.replace(tmp_output, output)
    return total_rows, left_rows


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists; pass --overwrite to replace it")

    logging.info("splits_dir=%s", args.splits_dir)
    logging.info("rollouts_dir=%s", args.rollouts_dir)
    logging.info("output=%s", args.output)

    done_ids = load_done_ids(args.rollouts_dir, args.valid_key, args.expected_rollouts)
    total_rows, left_rows = build_left_table(args.splits_dir, done_ids, args.output)
    logging.info(
        "done total_rows=%d done_ids=%d left_rows=%d output=%s",
        total_rows,
        len(done_ids),
        left_rows,
        args.output,
    )


if __name__ == "__main__":
    main()
