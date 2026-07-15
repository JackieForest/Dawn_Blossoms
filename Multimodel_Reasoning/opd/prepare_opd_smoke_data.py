#!/usr/bin/env python3
"""Create small per-domain parquet shards for an OPD smoke run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DOMAIN_FILES = [
    "train_chart_table_doc.parquet",
    "train_logic_game_puzzle.parquet",
    "train_math.parquet",
    "train_science.parquet",
    "train_spatial_general.parquet",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/mnt/petrelfs/linjuekai/Multimodel_Reasoning/rl_data/new_rl_data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/petrelfs/linjuekai/Multimodel_Reasoning/opd/data/smoke"),
    )
    parser.add_argument("--rows-per-domain", type=int, default=16)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for filename in DOMAIN_FILES:
        src = args.data_dir / filename
        if not src.exists():
            raise FileNotFoundError(src)

        parquet = pq.ParquetFile(src)
        remaining = args.rows_per_domain
        chunks: list[pa.Table] = []
        for row_group_idx in range(parquet.metadata.num_row_groups):
            if remaining <= 0:
                break
            table = parquet.read_row_group(row_group_idx)
            take = min(remaining, table.num_rows)
            chunks.append(table.slice(0, take))
            remaining -= take

        if not chunks:
            raise RuntimeError(f"No rows read from {src}")

        out = args.output_dir / filename
        pq.write_table(pa.concat_tables(chunks), out)
        written.append(out)
        print(f"wrote {out} rows={args.rows_per_domain - remaining}")

    list_file = args.output_dir / "train_files.txt"
    list_file.write_text("\n".join(str(path) for path in written) + "\n", encoding="utf-8")
    print(f"wrote {list_file}")


if __name__ == "__main__":
    main()
