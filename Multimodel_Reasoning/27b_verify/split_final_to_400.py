#!/usr/bin/env python3
"""Split filtered Final.parquet into ordered parquet shards."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


DEFAULT_SOURCE = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/"
    "rollouts/output/4bprfilter_split/Final.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/"
    "rollouts/output/qwen35_9b_distill/splits"
)


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "split_final_to_400.log"
    logger = logging.getLogger("split_final_to_400")
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


def split_sizes(total_rows: int, num_splits: int) -> list[int]:
    base = total_rows // num_splits
    rem = total_rows % num_splits
    return [base + (1 if i < rem else 0) for i in range(num_splits)]


def write_empty_split(path: Path, schema: pa.Schema, compression: str) -> None:
    pq.write_table(pa.Table.from_arrays([pa.array([], type=f.type) for f in schema], schema=schema), path, compression=compression)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split Final.parquet into ordered numeric parquet shards."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-splits", type=int, default=400)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip existing valid split files.")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"source parquet not found: {args.source}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir)

    existing = sorted([p for p in args.output_dir.glob("*.parquet") if p.stem.isdigit()])
    if existing and not (args.overwrite or args.resume):
        raise FileExistsError(
            f"{args.output_dir} already has numeric parquet files; use --overwrite or --resume"
        )
    if args.overwrite:
        for p in existing:
            p.unlink()

    source_pf = pq.ParquetFile(args.source)
    total_rows = source_pf.metadata.num_rows
    schema = source_pf.schema_arrow
    targets = split_sizes(total_rows, args.num_splits)

    logger.info("start split")
    logger.info("source=%s total_rows=%s", args.source, total_rows)
    logger.info("output_dir=%s num_splits=%s compression=%s", args.output_dir, args.num_splits, args.compression)

    split_summaries = []
    current_split = 0
    current_rows = 0
    writer: pq.ParquetWriter | None = None
    output_path: Path | None = None
    written_rows = [0 for _ in range(args.num_splits)]

    def open_writer(split_idx: int) -> pq.ParquetWriter:
        nonlocal output_path
        output_path = args.output_dir / f"{split_idx}.parquet"
        return pq.ParquetWriter(output_path, schema=schema, compression=args.compression)

    def close_writer() -> None:
        nonlocal writer, output_path
        if writer is not None:
            writer.close()
            assert output_path is not None
            logger.info(
                "wrote split=%s rows=%s path=%s",
                current_split,
                written_rows[current_split],
                output_path,
            )
            split_summaries.append(
                {
                    "split": current_split,
                    "rows": written_rows[current_split],
                    "path": str(output_path),
                }
            )
            writer = None
            output_path = None

    with tqdm(total=total_rows, desc="Splitting rows", unit="row") as pbar:
        for batch in source_pf.iter_batches(batch_size=8192):
            table = pa.Table.from_batches([batch], schema=schema)
            offset = 0
            while offset < table.num_rows:
                while current_split < args.num_splits and targets[current_split] == 0:
                    empty_path = args.output_dir / f"{current_split}.parquet"
                    write_empty_split(empty_path, schema, args.compression)
                    logger.info("wrote empty split=%s path=%s", current_split, empty_path)
                    split_summaries.append({"split": current_split, "rows": 0, "path": str(empty_path)})
                    current_split += 1
                    current_rows = 0

                if current_split >= args.num_splits:
                    raise RuntimeError("more rows than expected while splitting")

                if writer is None:
                    writer = open_writer(current_split)

                need = targets[current_split] - current_rows
                take = min(need, table.num_rows - offset)
                chunk = table.slice(offset, take)
                writer.write_table(chunk)

                offset += take
                current_rows += take
                written_rows[current_split] += take
                pbar.update(take)

                if current_rows == targets[current_split]:
                    close_writer()
                    current_split += 1
                    current_rows = 0

    if writer is not None:
        close_writer()

    while current_split < args.num_splits:
        empty_path = args.output_dir / f"{current_split}.parquet"
        write_empty_split(empty_path, schema, args.compression)
        logger.info("wrote trailing empty split=%s path=%s", current_split, empty_path)
        split_summaries.append({"split": current_split, "rows": 0, "path": str(empty_path)})
        current_split += 1

    final_files = [args.output_dir / f"{i}.parquet" for i in range(args.num_splits)]
    missing = [str(p) for p in final_files if not p.exists()]
    if missing:
        raise RuntimeError(f"missing output split files: {missing[:10]} total={len(missing)}")
    actual_rows = [pq.ParquetFile(p).metadata.num_rows for p in final_files]
    if actual_rows != targets:
        mismatches = [
            {"split": i, "expected": targets[i], "actual": actual_rows[i]}
            for i in range(args.num_splits)
            if targets[i] != actual_rows[i]
        ]
        raise RuntimeError(f"row count mismatch: {mismatches[:10]}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(args.source),
        "output_dir": str(args.output_dir),
        "num_splits": args.num_splits,
        "total_rows": total_rows,
        "min_split_rows": min(actual_rows) if actual_rows else 0,
        "max_split_rows": max(actual_rows) if actual_rows else 0,
        "splits": split_summaries,
    }
    summary_path = args.output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("done total_rows=%s summary=%s", total_rows, summary_path)

    print(f"Done. Wrote {args.num_splits} splits, total rows {total_rows}.")
    print(f"Rows per split: min={summary['min_split_rows']} max={summary['max_split_rows']}")
    print(f"Summary: {summary_path}")
    print(f"Log: {args.output_dir / 'split_final_to_400.log'}")


if __name__ == "__main__":
    main()
