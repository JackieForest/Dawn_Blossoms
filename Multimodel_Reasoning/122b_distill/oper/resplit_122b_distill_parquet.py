#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from tqdm import tqdm


DEFAULT_INPUT = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
    "qwen35_122b_distill/tmp_splits/122b_distill.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
    "qwen35_122b_distill/splits"
)


def target_rows(total_rows: int, num_splits: int) -> list[int]:
    base = total_rows // num_splits
    remainder = total_rows % num_splits
    return [base + (1 if idx < remainder else 0) for idx in range(num_splits)]


def numeric_parquet_files(directory: Path) -> list[Path]:
    return sorted(
        [p for p in directory.glob("*.parquet") if p.stem.isdigit()],
        key=lambda p: int(p.stem),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split merged 122B distill parquet into evenly sized split parquet files."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-splits", type=int, default=100)
    parser.add_argument("--expected-total", type=int, default=450315)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.num_splits <= 0:
        raise ValueError("--num-splits must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_outputs = numeric_parquet_files(args.output_dir)
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} already contains {len(existing_outputs)} numeric parquet files; "
            "pass --overwrite to replace them."
        )
    if args.overwrite:
        for path in existing_outputs:
            path.unlink()
        for path in args.output_dir.glob("*.tmp.*"):
            path.unlink()

    input_file = pq.ParquetFile(args.input)
    total_rows = input_file.metadata.num_rows
    schema = input_file.schema_arrow
    if args.expected_total >= 0 and total_rows != args.expected_total:
        raise RuntimeError(f"Expected input rows {args.expected_total}, got {total_rows}")

    per_split_rows = target_rows(total_rows, args.num_splits)
    started = time.time()
    split_summaries: list[dict[str, Any]] = []

    split_idx = 0
    written_current = 0
    writer: pq.ParquetWriter | None = None
    current_tmp_path: Path | None = None
    current_output_path: Path | None = None

    def open_writer(idx: int) -> pq.ParquetWriter:
        nonlocal current_tmp_path, current_output_path, written_current
        current_output_path = args.output_dir / f"{idx}.parquet"
        current_tmp_path = current_output_path.with_name(
            f"{current_output_path.name}.tmp.{idx}.{int(time.time())}"
        )
        if current_tmp_path.exists():
            current_tmp_path.unlink()
        written_current = 0
        return pq.ParquetWriter(
            current_tmp_path,
            schema,
            compression=args.compression,
            use_dictionary=False,
        )

    def close_writer(idx: int) -> None:
        nonlocal writer
        assert writer is not None
        assert current_tmp_path is not None
        assert current_output_path is not None
        writer.close()
        writer = None
        current_tmp_path.replace(current_output_path)

        actual_rows = pq.ParquetFile(current_output_path).metadata.num_rows
        expected_rows = per_split_rows[idx]
        if actual_rows != expected_rows:
            raise RuntimeError(
                f"Split {idx} wrote {actual_rows} rows, expected {expected_rows}"
            )
        split_summaries.append(
            {
                "split": idx,
                "path": str(current_output_path),
                "rows": actual_rows,
            }
        )

    try:
        writer = open_writer(split_idx)
        progress = tqdm(total=total_rows, desc="resplit parquet rows", dynamic_ncols=True)
        for batch in input_file.iter_batches(batch_size=max(1, args.batch_size)):
            offset = 0
            batch_rows = batch.num_rows
            while offset < batch_rows:
                if split_idx >= args.num_splits:
                    raise RuntimeError("Input has more rows than target split plan")

                remaining_for_split = per_split_rows[split_idx] - written_current
                take_rows = min(remaining_for_split, batch_rows - offset)
                if take_rows > 0:
                    assert writer is not None
                    writer.write_batch(batch.slice(offset, take_rows))
                    written_current += take_rows
                    offset += take_rows
                    progress.update(take_rows)

                if written_current == per_split_rows[split_idx]:
                    close_writer(split_idx)
                    split_idx += 1
                    if split_idx < args.num_splits:
                        writer = open_writer(split_idx)
        progress.close()

        if writer is not None:
            if written_current != per_split_rows[split_idx]:
                raise RuntimeError(
                    f"Final split {split_idx} incomplete: "
                    f"{written_current}/{per_split_rows[split_idx]}"
                )
            close_writer(split_idx)
            split_idx += 1
    except Exception:
        if writer is not None:
            writer.close()
        if current_tmp_path is not None and current_tmp_path.exists():
            current_tmp_path.unlink()
        raise

    output_files = numeric_parquet_files(args.output_dir)
    output_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in output_files)
    elapsed = time.time() - started

    if len(output_files) != args.num_splits:
        raise RuntimeError(f"Expected {args.num_splits} output files, found {len(output_files)}")
    if output_rows != total_rows:
        raise RuntimeError(f"Output row mismatch: output_rows={output_rows}, input_rows={total_rows}")
    if split_idx != args.num_splits:
        raise RuntimeError(f"Expected to write {args.num_splits} splits, wrote {split_idx}")

    summary = {
        "created_at": time.strftime("%F %T"),
        "elapsed_seconds": round(elapsed, 3),
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "num_splits": args.num_splits,
        "input_rows": total_rows,
        "output_files": len(output_files),
        "output_rows": output_rows,
        "expected_total": args.expected_total,
        "min_rows_per_split": min(per_split_rows) if per_split_rows else 0,
        "max_rows_per_split": max(per_split_rows) if per_split_rows else 0,
        "columns": schema.names,
        "split_summaries": split_summaries,
    }

    summary_json_path = args.output_dir / "RESPLIT_122B_DISTILL_SUMMARY.json"
    summary_txt_path = args.output_dir / "RESPLIT_122B_DISTILL_SUMMARY.txt"
    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_txt_path.write_text(
        "\n".join(
            [
                "122B distill parquet resplit summary",
                f"created_at: {summary['created_at']}",
                f"input: {args.input}",
                f"output_dir: {args.output_dir}",
                f"num_splits: {args.num_splits}",
                f"input_rows: {total_rows}",
                f"output_files: {len(output_files)}",
                f"output_rows: {output_rows}",
                f"expected_total: {args.expected_total}",
                f"min_rows_per_split: {summary['min_rows_per_split']}",
                f"max_rows_per_split: {summary['max_rows_per_split']}",
                f"columns: {len(schema.names)}",
                f"elapsed_seconds: {elapsed:.3f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(summary_txt_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
