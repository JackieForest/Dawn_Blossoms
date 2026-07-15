#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from tqdm import tqdm


DEFAULT_TMP_SPLITS = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
    "qwen35_122b_distill/tmp_splits"
)


def numeric_parquet_files(directory: Path) -> list[Path]:
    return sorted(
        [p for p in directory.glob("*.parquet") if p.stem.isdigit()],
        key=lambda p: int(p.stem),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge 400 qwen35_122b_distill tmp_splits parquet files into one parquet file."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_TMP_SPLITS)
    parser.add_argument("--output-name", default="122b_distill.parquet")
    parser.add_argument("--num-splits", type=int, default=400)
    parser.add_argument("--expected-total", type=int, default=450315)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_path = input_dir / args.output_name
    summary_json_path = input_dir / "MERGE_122B_DISTILL_SUMMARY.json"
    summary_txt_path = input_dir / "MERGE_122B_DISTILL_SUMMARY.txt"

    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    input_files = numeric_parquet_files(input_dir)
    if len(input_files) != args.num_splits:
        raise RuntimeError(
            f"Expected {args.num_splits} numeric parquet files in {input_dir}, found {len(input_files)}"
        )

    expected_paths = [input_dir / f"{idx}.parquet" for idx in range(args.num_splits)]
    missing = [str(path) for path in expected_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input splits: {missing[:10]}")

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")

    tmp_path = output_path.with_name(f"{output_path.name}.tmp.{int(time.time())}")
    if tmp_path.exists():
        tmp_path.unlink()

    started = time.time()
    split_summaries: list[dict[str, Any]] = []
    total_rows = 0
    writer: pq.ParquetWriter | None = None
    reference_schema = None

    try:
        for path in tqdm(expected_paths, desc="merge tmp_splits", dynamic_ncols=True):
            split_idx = int(path.stem)
            parquet_file = pq.ParquetFile(path)
            rows = parquet_file.metadata.num_rows
            row_groups = parquet_file.metadata.num_row_groups
            schema = parquet_file.schema_arrow

            if reference_schema is None:
                reference_schema = schema
                writer = pq.ParquetWriter(
                    tmp_path,
                    reference_schema,
                    compression=args.compression,
                    use_dictionary=False,
                )
            elif schema != reference_schema:
                raise RuntimeError(f"Schema mismatch at {path}")

            assert writer is not None
            for batch in parquet_file.iter_batches():
                writer.write_batch(batch)

            total_rows += rows
            split_summaries.append(
                {
                    "split": split_idx,
                    "path": str(path),
                    "rows": rows,
                    "row_groups": row_groups,
                }
            )
    except Exception:
        if writer is not None:
            writer.close()
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    else:
        if writer is not None:
            writer.close()

    tmp_path.replace(output_path)

    output_meta = pq.ParquetFile(output_path).metadata
    output_rows = output_meta.num_rows
    elapsed = time.time() - started

    if output_rows != total_rows:
        raise RuntimeError(f"Output row mismatch: output_rows={output_rows}, input_total={total_rows}")
    if args.expected_total >= 0 and output_rows != args.expected_total:
        raise RuntimeError(f"Expected total {args.expected_total}, got {output_rows}")

    summary = {
        "created_at": time.strftime("%F %T"),
        "elapsed_seconds": round(elapsed, 3),
        "input_dir": str(input_dir),
        "output_path": str(output_path),
        "num_splits": args.num_splits,
        "input_files": len(input_files),
        "expected_total": args.expected_total,
        "input_rows": total_rows,
        "output_rows": output_rows,
        "output_row_groups": output_meta.num_row_groups,
        "columns": reference_schema.names if reference_schema is not None else [],
        "split_summaries": split_summaries,
    }

    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_txt_path.write_text(
        "\n".join(
            [
                "122B tmp_splits merge summary",
                f"created_at: {summary['created_at']}",
                f"input_dir: {input_dir}",
                f"output_path: {output_path}",
                f"input_files: {len(input_files)}",
                f"num_splits: {args.num_splits}",
                f"input_rows: {total_rows}",
                f"output_rows: {output_rows}",
                f"expected_total: {args.expected_total}",
                f"output_row_groups: {output_meta.num_row_groups}",
                f"columns: {len(summary['columns'])}",
                f"elapsed_seconds: {elapsed:.3f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(summary_txt_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
