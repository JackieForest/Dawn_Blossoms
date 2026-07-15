#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


DEFAULT_27B_ROOT = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/qwen35_27b_distill"
)
DEFAULT_122B_ROOT = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/qwen35_122b_distill"
)


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad_lines = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                bad_lines += 1
                print(f"[WARN] skip bad JSON at {path}:{line_no}: {exc}", flush=True)
    return rows, bad_lines


def verify_key(row: dict[str, Any]) -> str:
    return str(row.get("id", row.get("index", "")))


def dataframe_keys(df: pd.DataFrame) -> pd.Series:
    if "id" in df.columns:
        return df["id"].astype(str)
    if "index" in df.columns:
        return df["index"].astype(str)
    raise KeyError("source split must contain either 'id' or 'index'")


def numeric_files(directory: Path, suffix: str) -> list[Path]:
    return sorted(
        [p for p in directory.glob(f"*{suffix}") if p.stem.isdigit()],
        key=lambda p: int(p.stem),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build 122B fallback temporary parquet splits by selecting 27B split rows "
            "whose verify_response correct_count is 0."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_27B_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_122B_ROOT)
    parser.add_argument("--num-splits", type=int, default=400)
    parser.add_argument("--expected-total", type=int, default=450315)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=128,
        help="Rows per parquet row group. Small groups avoid oversized image/text pages.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_splits = args.source_root / "splits"
    verify_dir = args.source_root / "verify_response"
    output_dir = args.output_root / "tmp_splits"
    summary_path = output_dir / "BUILD_SUMMARY.json"
    text_summary_path = output_dir / "BUILD_SUMMARY.txt"

    if not source_splits.exists():
        raise FileNotFoundError(source_splits)
    if not verify_dir.exists():
        raise FileNotFoundError(verify_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_outputs = numeric_files(output_dir, ".parquet")
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already contains {len(existing_outputs)} numeric parquet files; "
            "rerun with --overwrite to replace them."
        )
    if args.overwrite:
        for path in existing_outputs:
            path.unlink()
        for path in output_dir.glob("*.tmp.*"):
            path.unlink()

    started = time.time()
    totals = Counter()
    split_summaries: list[dict[str, Any]] = []

    for split_idx in tqdm(range(args.num_splits), desc="build tmp_splits", dynamic_ncols=True):
        split_path = source_splits / f"{split_idx}.parquet"
        verify_path = verify_dir / f"{split_idx}.jsonl"
        output_path = output_dir / f"{split_idx}.parquet"
        tmp_path = output_path.with_name(f"{output_path.name}.tmp.{split_idx}.{int(time.time())}")

        if not split_path.exists():
            raise FileNotFoundError(split_path)
        if not verify_path.exists():
            raise FileNotFoundError(verify_path)

        df = pd.read_parquet(split_path)
        verify_rows, bad_json_lines = load_jsonl(verify_path)

        totals["source_rows"] += len(df)
        totals["verify_rows"] += len(verify_rows)
        totals["bad_json_lines"] += bad_json_lines

        split_keys = dataframe_keys(df)
        split_key_list = split_keys.tolist()
        split_key_set = set(split_key_list)
        if len(split_keys) != len(set(split_keys)):
            raise ValueError(f"Duplicate source keys in {split_path}")

        verify_by_key: dict[str, dict[str, Any]] = {}
        for row in verify_rows:
            key = verify_key(row)
            if key in verify_by_key:
                raise ValueError(f"Duplicate verify key {key!r} in {verify_path}")
            verify_by_key[key] = row

        missing_verify = [key for key in split_key_list if key not in verify_by_key]
        extra_verify = [key for key in verify_by_key if key not in split_key_set]
        if missing_verify or extra_verify:
            raise ValueError(
                f"Key mismatch split {split_idx}: "
                f"missing_verify={len(missing_verify)} extra_verify={len(extra_verify)}"
            )

        cc_counts = Counter()
        invalid_rows = 0
        parse_failed_rows = 0
        selected_keys: set[str] = set()
        for key in split_key_list:
            verify_row = verify_by_key[key]
            if not verify_row.get("verify_valid"):
                invalid_rows += 1
                continue
            correct_count = int(verify_row.get("correct_count") or 0)
            cc_counts[correct_count] += 1
            if int(verify_row.get("parse_failed_count") or 0) > 0:
                parse_failed_rows += 1
            if correct_count == 0:
                selected_keys.add(key)

        selected_df = df.loc[split_keys.isin(selected_keys)].copy()
        if tmp_path.exists():
            tmp_path.unlink()
        selected_df.to_parquet(
            tmp_path,
            index=False,
            compression=args.compression,
            row_group_size=max(1, args.row_group_size),
        )
        tmp_path.replace(output_path)

        written_rows = pq.ParquetFile(output_path).metadata.num_rows
        if written_rows != len(selected_keys):
            raise RuntimeError(
                f"Split {split_idx} wrote {written_rows}, expected {len(selected_keys)}"
            )

        totals["selected_rows"] += written_rows
        totals["verify_valid_false"] += invalid_rows
        totals["parse_failed_rows"] += parse_failed_rows
        for key, value in cc_counts.items():
            totals[f"correct_count_{key}"] += value

        split_summaries.append(
            {
                "split": split_idx,
                "source_rows": len(df),
                "verify_rows": len(verify_rows),
                "bad_json_lines": bad_json_lines,
                "selected_rows": written_rows,
                "verify_valid_false": invalid_rows,
                "parse_failed_rows": parse_failed_rows,
                "correct_count": {str(k): cc_counts[k] for k in sorted(cc_counts)},
                "output": str(output_path),
            }
        )

    output_files = numeric_files(output_dir, ".parquet")
    output_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in output_files)
    elapsed = time.time() - started

    summary = {
        "created_at": time.strftime("%F %T"),
        "elapsed_seconds": round(elapsed, 3),
        "source_splits": str(source_splits),
        "verify_dir": str(verify_dir),
        "output_dir": str(output_dir),
        "num_splits": args.num_splits,
        "output_files": len(output_files),
        "expected_total": args.expected_total,
        "output_rows": output_rows,
        "totals": dict(totals),
        "split_summaries": split_summaries,
    }

    if len(output_files) != args.num_splits:
        raise RuntimeError(f"Expected {args.num_splits} output files, found {len(output_files)}")
    if output_rows != totals["selected_rows"]:
        raise RuntimeError(f"Output row mismatch: {output_rows} vs {totals['selected_rows']}")
    if args.expected_total >= 0 and output_rows != args.expected_total:
        raise RuntimeError(f"Expected total {args.expected_total}, got {output_rows}")

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text_summary_path.write_text(
        "\n".join(
            [
                "122B tmp_splits build summary",
                f"created_at: {summary['created_at']}",
                f"source_splits: {source_splits}",
                f"verify_dir: {verify_dir}",
                f"output_dir: {output_dir}",
                f"output_files: {len(output_files)}",
                f"output_rows: {output_rows}",
                f"expected_total: {args.expected_total}",
                f"source_rows: {totals['source_rows']}",
                f"verify_rows: {totals['verify_rows']}",
                f"bad_json_lines: {totals['bad_json_lines']}",
                f"correct_count_0: {totals['correct_count_0']}",
                f"correct_count_1: {totals['correct_count_1']}",
                f"correct_count_2: {totals['correct_count_2']}",
                f"correct_count_3: {totals['correct_count_3']}",
                f"correct_count_4: {totals['correct_count_4']}",
                f"verify_valid_false: {totals['verify_valid_false']}",
                f"parse_failed_rows: {totals['parse_failed_rows']}",
                f"elapsed_seconds: {elapsed:.3f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(text_summary_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
