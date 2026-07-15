#!/usr/bin/env python3
"""Filter original parquet splits by 27B verify correct_count.

Keeps samples whose 4 rollout judgments have correct_count in {0, 1, 2}.
The output parquet files keep the same columns/schema as the source split files.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm


DEFAULT_SOURCE_SPLIT_DIR = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/"
    "rollouts/output/qwen35_4b_pr/splits"
)
DEFAULT_VERIFY_DIR = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/"
    "rollouts/output/qwen35_27b_verify/verify"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/"
    "rollouts/output/4bprfilter_split"
)


def numeric_jsonl_files(path: Path) -> list[Path]:
    return sorted(
        [p for p in path.glob("*.jsonl") if p.stem.isdigit()],
        key=lambda p: int(p.stem),
    )


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "filter.log"
    logger = logging.getLogger("filter_parquet_by_verify")
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


def read_keep_indexes(
    verify_file: Path,
    keep_counts: set[int],
    expected_rollouts: int,
) -> tuple[set[int], Counter, int]:
    keep_indexes: set[int] = set()
    dist: Counter[int] = Counter()
    invalid = 0

    with verify_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad json in {verify_file}:{line_no}: {exc}") from exc

            if not obj.get("verify_valid") or int(obj.get("num_rollouts", -1)) != expected_rollouts:
                invalid += 1
                continue

            correct_count = int(obj.get("correct_count", -1))
            dist[correct_count] += 1
            if correct_count in keep_counts:
                keep_indexes.add(int(obj["index"]))

    return keep_indexes, dist, invalid


def filter_one_split(
    source_file: Path,
    output_file: Path,
    keep_indexes: set[int],
    compression: str,
) -> tuple[int, int]:
    table = pq.read_table(source_file)
    source_rows = table.num_rows

    if not keep_indexes:
        filtered = table.slice(0, 0)
    else:
        keep_array = pa.array(sorted(keep_indexes), type=pa.int64())
        mask = pc.is_in(table["index"], value_set=keep_array)
        filtered = table.filter(mask)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(filtered, output_file, compression=compression)
    return source_rows, filtered.num_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter qwen35_4b_pr parquet splits by 27B verify correct_count."
    )
    parser.add_argument("--source-split-dir", type=Path, default=DEFAULT_SOURCE_SPLIT_DIR)
    parser.add_argument("--verify-dir", type=Path, default=DEFAULT_VERIFY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--keep-counts",
        default="0,1,2",
        help="Comma-separated correct_count values to keep. Default: 0,1,2.",
    )
    parser.add_argument("--expected-rollouts", type=int, default=4)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing output parquet files.",
    )
    args = parser.parse_args()

    keep_counts = {int(x) for x in args.keep_counts.split(",") if x.strip()}
    logger = setup_logger(args.output_dir)

    verify_files = numeric_jsonl_files(args.verify_dir)
    if not verify_files:
        raise SystemExit(f"no numeric verify jsonl files found in {args.verify_dir}")

    logger.info("start filter")
    logger.info("source_split_dir=%s", args.source_split_dir)
    logger.info("verify_dir=%s", args.verify_dir)
    logger.info("output_dir=%s", args.output_dir)
    logger.info("keep_counts=%s expected_rollouts=%s", sorted(keep_counts), args.expected_rollouts)

    total_source_rows = 0
    total_kept_rows = 0
    total_invalid_verify = 0
    total_verify_dist: Counter[int] = Counter()
    split_summaries = []

    progress = tqdm(verify_files, desc="Filtering splits", unit="split")
    for verify_file in progress:
        split_idx = int(verify_file.stem)
        source_file = args.source_split_dir / f"{split_idx}.parquet"
        output_file = args.output_dir / f"{split_idx}.parquet"

        if not source_file.exists():
            raise FileNotFoundError(f"missing source split: {source_file}")
        if output_file.exists() and not args.overwrite:
            if args.resume:
                rows = pq.ParquetFile(output_file).metadata.num_rows
                progress.set_postfix(split=split_idx, kept=rows, skipped=True)
                logger.info("skip existing split=%s output=%s rows=%s", split_idx, output_file, rows)
                continue
            raise FileExistsError(f"{output_file} exists; use --overwrite or --resume")

        keep_indexes, verify_dist, invalid_verify = read_keep_indexes(
            verify_file,
            keep_counts=keep_counts,
            expected_rollouts=args.expected_rollouts,
        )
        source_rows, kept_rows = filter_one_split(
            source_file,
            output_file,
            keep_indexes=keep_indexes,
            compression=args.compression,
        )

        total_source_rows += source_rows
        total_kept_rows += kept_rows
        total_invalid_verify += invalid_verify
        total_verify_dist.update(verify_dist)
        split_summaries.append(
            {
                "split": split_idx,
                "source_rows": source_rows,
                "kept_rows": kept_rows,
                "invalid_verify_rows": invalid_verify,
                "verify_distribution": {str(k): verify_dist[k] for k in sorted(verify_dist)},
                "output_file": str(output_file),
            }
        )

        progress.set_postfix(split=split_idx, kept=kept_rows)
        logger.info(
            "split=%s source_rows=%s kept_rows=%s invalid_verify=%s dist=%s output=%s",
            split_idx,
            source_rows,
            kept_rows,
            invalid_verify,
            dict(sorted(verify_dist.items())),
            output_file,
        )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_split_dir": str(args.source_split_dir),
        "verify_dir": str(args.verify_dir),
        "output_dir": str(args.output_dir),
        "keep_counts": sorted(keep_counts),
        "expected_rollouts": args.expected_rollouts,
        "num_verify_files": len(verify_files),
        "total_source_rows": total_source_rows,
        "total_kept_rows": total_kept_rows,
        "kept_rate": total_kept_rows / total_source_rows if total_source_rows else 0.0,
        "total_invalid_verify_rows": total_invalid_verify,
        "verify_distribution": {str(k): total_verify_dist[k] for k in sorted(total_verify_dist)},
        "splits": split_summaries,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    logger.info("done total_source_rows=%s total_kept_rows=%s kept_rate=%.6f", total_source_rows, total_kept_rows, summary["kept_rate"])
    logger.info("summary=%s", summary_path)
    print(f"Done. kept {total_kept_rows}/{total_source_rows} ({summary['kept_rate'] * 100:.2f}%)")
    print(f"Summary: {summary_path}")
    print(f"Log: {args.output_dir / 'filter.log'}")


if __name__ == "__main__":
    main()
