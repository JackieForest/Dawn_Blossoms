#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


DEFAULT_FINAL_SPLITS_DIR = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
    "distill_results/9b_distill/final_splits"
)
DEFAULT_FINAL_ROLLOUTS_DIR = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
    "distill_results/9b_distill/final_rollouts"
)
DEFAULT_OUTPUT_DIR = DEFAULT_FINAL_SPLITS_DIR / "final"
DEFAULT_OUTPUT_NAME = "Final_qwen35_9b_distill-360k.parquet"
DEFAULT_LOG_FILE = Path(
    "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/rollouts/output/"
    "distill_results/9b_distill/build_final_9b_distill_parquet.log"
)


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_rollout_responses(path: Path) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    responses: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("id", row.get("index", "")))
            if not sid:
                raise RuntimeError(f"missing id in {path}:{line_no}")
            response = row.get("response")
            if response is None:
                raise RuntimeError(f"missing response for id={sid} in {path}:{line_no}")
            ids.append(sid)
            responses.append(str(response))
    return ids, responses


def write_table(writer: pq.ParquetWriter | None, output_tmp: Path, df: pd.DataFrame) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(output_tmp, table.schema, compression="snappy")
    elif table.schema != writer.schema:
        table = table.cast(writer.schema)
    writer.write_table(table)
    return writer


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build one final 9B distill parquet by adding the selected best rollout response "
            "from final_rollouts to final_splits."
        )
    )
    parser.add_argument("--final-splits-dir", default=str(DEFAULT_FINAL_SPLITS_DIR))
    parser.add_argument("--final-rollouts-dir", default=str(DEFAULT_FINAL_ROLLOUTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--num-splits", type=int, default=400)
    parser.add_argument("--log-file", default=str(DEFAULT_LOG_FILE))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    setup_logging(Path(args.log_file))

    final_splits_dir = Path(args.final_splits_dir)
    final_rollouts_dir = Path(args.final_rollouts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    output_tmp = output_path.with_name(f"{output_path.name}.tmp")

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite")
    if output_tmp.exists():
        raise FileExistsError(f"temporary file exists from a previous run: {output_tmp}")

    logging.info("final_splits_dir=%s", final_splits_dir)
    logging.info("final_rollouts_dir=%s", final_rollouts_dir)
    logging.info("output_path=%s", output_path)
    logging.info("num_splits=%s overwrite=%s", args.num_splits, args.overwrite)

    writer: pq.ParquetWriter | None = None
    seen_ids: set[str] = set()
    total_rows = 0

    try:
        for split in tqdm(range(args.num_splits), desc="build final parquet", dynamic_ncols=True):
            split_path = final_splits_dir / f"{split}.parquet"
            rollout_path = final_rollouts_dir / f"{split}.jsonl"
            if not split_path.exists():
                raise FileNotFoundError(split_path)
            if not rollout_path.exists():
                raise FileNotFoundError(rollout_path)

            df = pd.read_parquet(split_path)
            split_ids = df["id"].astype(str).tolist()
            rollout_ids, responses = load_rollout_responses(rollout_path)

            if split_ids != rollout_ids:
                if len(split_ids) != len(rollout_ids):
                    raise RuntimeError(
                        f"split {split}: row count mismatch final_splits={len(split_ids)} "
                        f"final_rollouts={len(rollout_ids)}"
                    )
                for row_idx, (left, right) in enumerate(zip(split_ids, rollout_ids)):
                    if left != right:
                        raise RuntimeError(
                            f"split {split}: id mismatch at row {row_idx}: "
                            f"final_splits id={left!r}, final_rollouts id={right!r}"
                        )

            repeated = [sid for sid in split_ids if sid in seen_ids]
            if repeated:
                raise RuntimeError(f"split {split}: duplicate ids across splits, preview={repeated[:10]}")
            seen_ids.update(split_ids)

            if "response" in df.columns:
                raise RuntimeError(f"split {split}: final_splits already has a response column")
            df["response"] = responses

            writer = write_table(writer, output_tmp, df)
            total_rows += len(df)
            logging.info("wrote split=%s rows=%s total_rows=%s", split, len(df), total_rows)

        if writer is not None:
            writer.close()
            writer = None
        output_tmp.replace(output_path)
        logging.info("done total_rows=%s unique_ids=%s output=%s", total_rows, len(seen_ids), output_path)
    except Exception:
        if writer is not None:
            writer.close()
        logging.exception("failed; partial tmp remains at %s", output_tmp)
        raise


if __name__ == "__main__":
    main()
