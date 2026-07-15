#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import aiohttp
from tqdm import tqdm

from common import load_config, load_prompt, truncate_text


def render_verify_prompt(template: str, question: str, solution: str, response: str) -> str:
    replacements = {
        "{question}": truncate_text(question),
        "{solution}": truncate_text(solution),
        "{answer}": truncate_text(solution),
        "{response}": truncate_text(response, max_chars=30000),
    }
    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt.strip()


def normalize_judgment(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    lowered = raw.lower()
    words = re.findall(r"[a-z]+", lowered)
    if words:
        if words[0] == "correct":
            return "correct", raw
        if words[0] == "wrong":
            return "wrong", raw
    if "correct" in lowered and "wrong" not in lowered:
        return "correct", raw
    return "wrong", raw


def load_existing_indexes(path: Path, valid_key: str, expected_rollouts: int) -> set[int]:
    if not path.exists():
        return set()
    existing: set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            judgments = obj.get("judgments", [])
            if obj.get(valid_key) and "index" in obj and isinstance(judgments, list) and len(judgments) == expected_rollouts:
                existing.add(int(obj["index"]))
    return existing


def iter_input_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                records.append(
                    {
                        "index": -line_no,
                        "verify_valid": False,
                        "error": f"bad input json line {line_no}: {exc}",
                    }
                )
                continue
            records.append(obj)
    return records


def metadata_record(example: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "index",
        "id",
        "source",
        "original_id",
        "domain",
        "type",
        "subtype",
        "question",
        "answer",
    ]
    return {key: example.get(key) for key in keys if key in example}


async def request_one_judgment(
    session: aiohttp.ClientSession,
    api_url: str,
    cfg: dict[str, Any],
    prompt: str,
    rollout: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg.get("temperature", 0.7),
        "top_p": cfg.get("top_p", 0.8),
        "max_tokens": cfg.get("max_tokens", 8),
    }
    for key in ("top_k", "min_p", "presence_penalty", "repetition_penalty"):
        if key in cfg:
            payload[key] = cfg[key]
    if "enable_thinking" in cfg:
        payload["chat_template_kwargs"] = {"enable_thinking": bool(cfg.get("enable_thinking"))}

    max_retries = int(cfg.get("max_retries", 3))
    last_error = ""
    for attempt in range(max_retries):
        try:
            async with session.post(f"{api_url}/chat/completions", json=payload) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status}: {text[:1000]}")
                result = json.loads(text)
            choice = result["choices"][0]
            judge_response = choice["message"]["content"]
            judgment, raw = normalize_judgment(judge_response)
            return {
                "rollout_id": rollout.get("rollout_id"),
                "judgment": judgment,
                "judge_response": raw,
                "finish_reason": choice.get("finish_reason", ""),
            }
        except Exception as exc:
            last_error = repr(exc)
            if attempt < max_retries - 1:
                await asyncio.sleep(2 * (attempt + 1))

    return {
        "rollout_id": rollout.get("rollout_id"),
        "judgment": "wrong",
        "judge_response": "",
        "finish_reason": "",
        "error": last_error,
    }


async def verify_one_sample(
    example: dict[str, Any],
    session: aiohttp.ClientSession,
    api_url: str,
    cfg: dict[str, Any],
    prompt_template: str,
) -> dict[str, Any]:
    record = metadata_record(example)
    rollouts = example.get("rollouts", [])
    expected_rollouts = int(cfg.get("expected_rollouts", cfg.get("rollouts_per_sample", 4)))
    if not example.get("rollout_valid") or not isinstance(rollouts, list) or len(rollouts) != expected_rollouts:
        return {
            **record,
            "verify_valid": False,
            "error": "input rollout record is invalid or incomplete",
            "input_num_rollouts": len(rollouts) if isinstance(rollouts, list) else 0,
        }

    question = str(example.get("question", ""))
    solution = str(example.get("answer", ""))
    tasks = []
    for rollout in rollouts:
        response = str(rollout.get("response", ""))
        prompt = render_verify_prompt(prompt_template, question, solution, response)
        tasks.append(asyncio.create_task(request_one_judgment(session, api_url, cfg, prompt, rollout)))
    judgments = await asyncio.gather(*tasks)

    for item in judgments:
        if item.get("error"):
            return {
                **record,
                "verify_valid": False,
                "judgments": [j for j in judgments if not j.get("error")],
                "error": item["error"],
                "failed_rollout_id": item.get("rollout_id"),
            }

    correct_count = sum(1 for item in judgments if item.get("judgment") == "correct")
    return {
        **record,
        "verify_valid": True,
        "num_rollouts": expected_rollouts,
        "correct_count": correct_count,
        "wrong_count": expected_rollouts - correct_count,
        "judgments": judgments,
    }


async def process_split(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    input_dir = Path(cfg["input_rollout_dir"])
    output_dir = Path(cfg["output_dir"])
    success_name = str(cfg.get("success_dir_name", "verify"))
    failed_name = str(cfg.get("failed_dir_name", "failed_verify"))
    valid_key = str(cfg.get("valid_key", "verify_valid"))
    jsonl_dir = output_dir / success_name
    failed_dir = output_dir / failed_name
    stop_dir = output_dir / "stop_files"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    stop_dir.mkdir(parents=True, exist_ok=True)

    input_file = input_dir / f"{args.index}.jsonl"
    output_file = jsonl_dir / f"{args.index}.jsonl"
    failed_file = failed_dir / f"{args.index}.jsonl"
    prompt_template = load_prompt(cfg)

    print(f"[{cfg.get('name', '?')}] Loading rollout split {args.index}: {input_file}", flush=True)
    records = iter_input_records(input_file)
    if not input_file.exists():
        failed_file.open("a", encoding="utf-8").write(
            json.dumps({"index": args.index, "verify_valid": False, "error": f"missing input file: {input_file}"}, ensure_ascii=False) + "\n"
        )
        raise FileNotFoundError(input_file)

    expected_rollouts = int(cfg.get("expected_rollouts", cfg.get("rollouts_per_sample", 4)))
    existing = load_existing_indexes(output_file, valid_key, expected_rollouts)
    work = [obj for obj in records if int(obj.get("index", -1)) not in existing]
    print(f"rows={len(records)} done={len(existing)} remaining={len(work)}", flush=True)

    if not work:
        (stop_dir / f"verify_{args.index}.flag").write_text("stop\n", encoding="utf-8")
        print("Split already complete; wrote stop flag.", flush=True)
        return

    api_url = f"http://{args.url}:{args.port}/v1"
    timeout = aiohttp.ClientTimeout(total=int(cfg.get("request_timeout", 300)))
    max_concurrent = int(cfg.get("max_concurrent", 64))
    max_pending = max_concurrent * int(cfg.get("max_pending_multiplier", 2))

    ok = 0
    fail = 0
    started = time.time()
    iterator = iter(work)
    active: set[asyncio.Task] = set()

    def schedule(session: aiohttp.ClientSession) -> bool:
        try:
            example = next(iterator)
        except StopIteration:
            return False
        task = asyncio.create_task(verify_one_sample(example, session, api_url, cfg, prompt_template))
        active.add(task)
        return True

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for _ in range(min(max_pending, len(work))):
            schedule(session)

        with output_file.open("a", encoding="utf-8") as out_f, failed_file.open("a", encoding="utf-8") as fail_f:
            pbar = tqdm(total=len(work), desc=f"verify split {args.index}", unit="sample")
            while active:
                done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    record = task.result()
                    if record.get(valid_key):
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_f.flush()
                        ok += 1
                    else:
                        fail_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fail_f.flush()
                        fail += 1
                    pbar.update(1)
                    pbar.set_postfix(ok=ok, fail=fail)
                    schedule(session)
            pbar.close()

    elapsed = time.time() - started
    print(f"Done verify split {args.index}: ok={ok} fail={fail} elapsed={elapsed:.1f}s", flush=True)
    if len(load_existing_indexes(output_file, valid_key, expected_rollouts)) >= len(records):
        (stop_dir / f"verify_{args.index}.flag").write_text("stop\n", encoding="utf-8")
        print("Split complete; wrote stop flag.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3.5-27B judge verification for one 4B rollout split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--url", required=True, help="vLLM server host without port.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ[key] = ""

    cfg = load_config(args.config)
    asyncio.run(process_split(args, cfg))


if __name__ == "__main__":
    main()
