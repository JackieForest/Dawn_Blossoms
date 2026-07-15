#!/usr/bin/env python3
"""Merge filtered parquet splits into one Final.parquet, then remove split files."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


DEFAULT_INPUT_DIR = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/"
    "rollouts/output/4bprfilter_split"
)


def setup_logger(input_dir: Path) -> logging.Logger:
    input_dir.mkdir(parents=True, exist_ok=True)
    log_path = input_dir / "merge_final.log"
    logger = logging.getLogger("merge_filter_splits")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def numeric_split_files(input_dir: Path) -> list[Path]:
    return sorted(
        [p for p in input_dir.glob("*.parquet") if p.stem.isdigit()],
        key=lambda p: int(p.stem),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge 0.parquet...99.parquet into Final.parquet and delete split files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-name", default="Final.parquet")
    parser.add_argument("--expected-splits", type=int, default=100)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-splits",
        action="store_true",
        help="Do not delete numeric split parquet files after successful merge.",
    )
    args = parser.parse_args()

    logger = setup_logger(args.input_dir)
    output_path = args.input_dir / args.output_name
    split_files = numeric_split_files(args.input_dir)

    logger.info("start merge")
    logger.info("input_dir=%s", args.input_dir)
    logger.info("output_path=%s", output_path)
    logger.info("expected_splits=%s compression=%s", args.expected_splits, args.compression)

    expected_names = {f"{i}.parquet" for i in range(args.expected_splits)}
    found_names = {p.name for p in split_files}
    missing = sorted(expected_names - found_names, key=lambda x: int(Path(x).stem))
    extra = sorted(found_names - expected_names, key=lambda x: int(Path(x).stem))
    if missing:
        raise FileNotFoundError(f"missing split files: {missing[:20]} total={len(missing)}")
    if extra:
        logger.warning("extra numeric parquet files outside expected range: %s", extra)

    split_files = [args.input_dir / f"{i}.parquet" for i in range(args.expected_splits)]
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_path} exists; use --overwrite")
        logger.info("remove existing output before overwrite: %s", output_path)
        output_path.unlink()

    source_rows = {}
    expected_rows = 0
    schema = None
    writer = None
    try:
        for split_file in tqdm(split_files, desc="Merging parquet", unit="split"):
            parquet_file = pq.ParquetFile(split_file)
            rows = parquet_file.metadata.num_rows
            source_rows[split_file.name] = rows
            expected_rows += rows

            if schema is None:
                schema = parquet_file.schema_arrow
                writer = pq.ParquetWriter(
                    output_path,
                    schema=schema,
                    compression=args.compression,
                )
            elif parquet_file.schema_arrow != schema:
                raise ValueError(f"schema mismatch: {split_file}")

            assert writer is not None
            for rg_idx in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(rg_idx)
                writer.write_table(table)

            logger.info("merged split=%s rows=%s", split_file.name, rows)
    finally:
        if writer is not None:
            writer.close()

    final_rows = pq.ParquetFile(output_path).metadata.num_rows
    if final_rows != expected_rows:
        raise RuntimeError(
            f"merged row count mismatch: Final.parquet={final_rows}, expected={expected_rows}"
        )

    deleted = []
    if not args.keep_splits:
        for split_file in tqdm(split_files, desc="Deleting splits", unit="file"):
            split_file.unlink()
            deleted.append(split_file.name)
            logger.info("deleted split=%s", split_file)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(args.input_dir),
        "output_path": str(output_path),
        "expected_splits": args.expected_splits,
        "final_rows": final_rows,
        "source_rows": source_rows,
        "deleted_splits": deleted,
        "kept_splits": bool(args.keep_splits),
    }
    summary_path = args.input_dir / "merge_final_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    logger.info("done final_rows=%s output=%s summary=%s", final_rows, output_path, summary_path)
    print(f"Done. Final rows: {final_rows}")
    print(f"Output: {output_path}")
    print(f"Summary: {summary_path}")
    print(f"Log: {args.input_dir / 'merge_final.log'}")


if __name__ == "__main__":
    main()
