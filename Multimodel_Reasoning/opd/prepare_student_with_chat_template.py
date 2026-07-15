#!/usr/bin/env python3
"""Create a lightweight model directory with a chat template added.

The base Qwen3.5 model directory may not include chat_template.jinja. verl's
multimodal RL dataset calls processor.apply_chat_template(), so the processor
needs a chat template during prompt filtering. This script symlinks the base
model files into an OPD-local directory and copies a known-good template.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("/mnt/dhwfile/raise/user/linjuekai/models/Qwen3.5-4B-Base"),
    )
    parser.add_argument(
        "--chat-template",
        type=Path,
        default=Path(
            "/mnt/dhwfile/raise/user/linjuekai/Multimodel_Reasoning/training/models/"
            "qwen35_4b_base_full_distill_sft_528k_16k/chat_template.jinja"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/petrelfs/linjuekai/Multimodel_Reasoning/opd/models/Qwen3.5-4B-Base-chat-template"),
    )
    args = parser.parse_args()

    if not args.base_model.is_dir():
        raise FileNotFoundError(args.base_model)
    if not args.chat_template.is_file():
        raise FileNotFoundError(args.chat_template)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for src in args.base_model.iterdir():
        dst = args.output_dir / src.name
        if dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(src)

    template_dst = args.output_dir / "chat_template.jinja"
    template_dst.write_text(args.chat_template.read_text(encoding="utf-8"), encoding="utf-8")

    print(args.output_dir)


if __name__ == "__main__":
    main()
