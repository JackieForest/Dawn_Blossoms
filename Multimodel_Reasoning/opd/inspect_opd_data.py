#!/usr/bin/env python3
"""Inspect RL parquet files before OPD smoke runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/mnt/petrelfs/linjuekai/Multimodel_Reasoning/rl_data/new_rl_data"),
    )
    parser.add_argument("--pattern", default="train_*.parquet")
    args = parser.parse_args()

    files = sorted(args.data_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No parquet files found under {args.data_dir} with pattern {args.pattern!r}")

    for path in files:
        parquet = pq.ParquetFile(path)
        print(f"\n{path}")
        print(f"  rows: {parquet.metadata.num_rows}")
        print(f"  columns: {', '.join(parquet.schema_arrow.names)}")
        if "data_source" in parquet.schema_arrow.names:
            table = parquet.read(columns=["data_source"])
            counts = pc.value_counts(table["data_source"]).to_pylist()
            print(f"  data_source: {counts}")
        if "extra_info" in parquet.schema_arrow.names:
            table = parquet.read(columns=["extra_info"])
            extra = table["extra_info"].to_pylist()
            domains: dict[str, int] = {}
            for item in extra:
                domain = item.get("domain") if isinstance(item, dict) else None
                domains[str(domain)] = domains.get(str(domain), 0) + 1
            print(f"  extra_info.domain: {domains}")


if __name__ == "__main__":
    main()
