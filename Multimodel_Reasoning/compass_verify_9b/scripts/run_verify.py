#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from common import as_bool, load_config


CV_COT_PROMPT = """As a grading expert, your task is to determine whether the candidate's final answer matches the provided standard answer. Follow these evaluation guidelines precisely:

Evaluation Protocol:
1. Reference Standard:
   - The standard answer is definitive and always correct
   - The question is perfectly valid - never question them
   - Do not regenerate answers; only compare with the given standard

2. Comparison Method:
   - Carefully analyze the question's requirements and the standard answer's structure
     * Determine whether the question expects exact matching of the entire standard answer or allows partial matching of its components.
     * This determination must be made based on the question's phrasing and the nature of the standard answer.
   - Compare ONLY the candidate's final answer (ignore all reasoning/explanation errors)
   - Disregard any differences in formatting or presentation style
   - For mathematical expressions: calculate step by step whether the two formulas are equivalent
   - For multiple-choice questions: compare only the final choice and corresponding option content

3. Multi-part Answers:
   - For questions requiring multiple responses (e.g., multi-select):
   - All parts must match the standard answer exactly.
   - Compare each sub-answer step by step. Partial matches are considered incorrect.

4. Validity Check:
   - Reject answers that are:
     * Incomplete (cut off mid-sentence in the final sentence, lacking a complete response) -> Label as INCOMPLETE
     * Repetitive (repetition of words or phrases in a loop) -> Label as REPETITIVE
     * Explicit refusals (e.g., directly return "I cannot answer/provide/access ...") -> Label as REFUSAL
   - For invalid answers, specify the type in the judgment (e.g., \\boxed{{C}} - INCOMPLETE).

Grading Scale:
\\boxed{{A}} - CORRECT:
   - Answer matches standard exactly (including equivalent expressions)
   - For numerical answers: consider as equivalent if values match when rounded appropriately
   - Semantically equivalent responses

\\boxed{{B}} - INCORRECT:
   - Any deviation from standard answer
   - Partial matches for multi-part questions

\\boxed{{C}} - INCOMPLETE/REPETITIVE/REFUSAL:
   - Fails validity criteria above (must specify: INCOMPLETE/REPETITIVE/REFUSAL)

Execution Steps and Output Formats:

Analysis step by step: [
Thoroughly evaluate the candidate's answer including:
(1) First check if the answer is INCOMPLETE (cut off mid-sentence), REPETITIVE (looping repetition), or a REFUSAL (explicit denial) - if so, immediately classify as \\boxed{{C}} with the corresponding type.
(2) Analyze the question's core requirements and the standard answer's structure, for example:
- Strict requirements: Identify mandatory constraints (e.g., simplification, answer order, multi-part completeness)
- Tolerant allowances: Ignore non-critical deviations (e.g., missing option labels in MCQs, equivalent but unformatted expressions)
- Required answer type, precision level, etc.
(3) Perform a detailed comparison between the candidate's final answer and the standard answer, for example:
- Content equivalence
- Permitted variations in numerical precision
- Allowed expression formats]
Final Judgment: \\boxed{{A/B/C}} - <CORRECT/INCORRECT/INCOMPLETE/REPETITIVE/REFUSAL>

Here is your task.
<Original Question Begin>
{question}
<Original Question End>

<Standard Answer Begin>
{gold_answer}
<Standard Answer End>

<Candidate's Answer Begin>
{llm_response}
<Candidate's Answer End>

Analysis step by step and Final Judgment:
"""

SYSTEM_PROMPT = "Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly."


def truncate(value: Any, max_chars: int = 20000) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= max_chars else text[:max_chars] + "\n[TRUNCATED]"


def process_judgment(text: str) -> str:
    boxed = re.findall(r"boxed\{([A-C])\}", text)
    if boxed:
        return boxed[-1]
    stripped = text.strip()
    if stripped in {"A", "B", "C"}:
        return stripped
    final = stripped.split("Final Judgment:")[-1]
    matches = re.findall(r"\b([A-C])\b", final)
    return matches[-1] if matches else ""


def check_rollout_format(response: Any) -> tuple[bool, str]:
    text = "" if response is None else str(response).strip()
    if not re.search(r"<think>.*?</think>", text, flags=re.DOTALL | re.IGNORECASE):
        return False, "missing_think_tag"
    if not re.search(r"<answer>.*?</answer>", text, flags=re.DOTALL | re.IGNORECASE):
        return False, "missing_answer_tag"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False, "empty_response"
    last_line = lines[-1]
    if not re.fullmatch(r"Therefore,\s+the\s+final\s+answer\s+is\s+<answer>.+?</answer>", last_line, flags=re.DOTALL):
        return False, "bad_final_line"
    return True, ""


def format_prompt(tokenizer: AutoTokenizer, question: str, answer: str, candidate: str) -> str:
    prompt = CV_COT_PROMPT.format(
        question=truncate(question),
        gold_answer=truncate(answer),
        llm_response=truncate(candidate),
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if tokenizer.bos_token and formatted.startswith(tokenizer.bos_token):
        formatted = formatted.removeprefix(tokenizer.bos_token)
    return formatted


def record_key(item: dict[str, Any]) -> str:
    return str(item.get("id", item.get("index", "")))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] skip bad json {path}:{line_no}: {exc}", flush=True)
    return rows


def dump_jsonl_line(f, obj: dict[str, Any]) -> None:
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    f.flush()


def build_tasks(item: dict[str, Any], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rollouts = item.get("rollouts")
    if not isinstance(rollouts, list):
        rollouts = []
    tasks: list[dict[str, Any]] = []
    candidates: list[str] = []
    for ridx, rollout in enumerate(rollouts):
        if not isinstance(rollout, dict):
            rollout = {}
        candidate = rollout.get("response", "")
        candidate = str(candidate)
        format_valid, format_error = check_rollout_format(rollout.get("response", ""))
        candidates.append(candidate)
        tasks.append(
            {
                "rollout_id": rollout.get("rollout_id", ridx),
                "candidate": candidate,
                "final_answer": rollout.get("final_answer"),
                "format_valid": format_valid,
                "format_error": format_error,
                "finish_reason": rollout.get("finish_reason"),
                "response_chars": rollout.get("response_chars"),
                "has_answer_tag": rollout.get("has_answer_tag"),
            }
        )
    return tasks, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify 9B distill rollouts with CompassVerifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    index = args.index
    input_dir = Path(cfg["input_rollout_dir"])
    output_root = Path(cfg["output_dir"])
    verify_dir = output_root / str(cfg.get("success_dir_name", "verify"))
    failed_dir = output_root / str(cfg.get("failed_dir_name", "failed_verify"))
    stop_dir = output_root / str(cfg.get("stop_dir_name", "verify_stop_files"))
    verify_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    stop_dir.mkdir(parents=True, exist_ok=True)

    input_path = input_dir / f"{index}.jsonl"
    output_path = verify_dir / f"{index}.jsonl"
    failed_path = failed_dir / f"{index}.jsonl"
    stop_path = stop_dir / f"verify_{index}.flag"

    if stop_path.exists():
        print(f"stop flag exists: {stop_path}", flush=True)
        return
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    all_items = load_jsonl(input_path)
    done: dict[str, dict[str, Any]] = {}
    for item in load_jsonl(output_path):
        if item.get("verify_valid") and item.get("num_rollouts"):
            done[record_key(item)] = item
    todo = [item for item in all_items if record_key(item) not in done]

    print(f"split={index} input={len(all_items)} cached={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        stop_path.write_text(f"done {time.strftime('%F %T')}\n", encoding="utf-8")
        return

    model_path = str(cfg["verifier_model"])
    tp = int(cfg.get("vllm", {}).get("tensor_parallel_size", 1))
    model = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tp,
        max_model_len=int(cfg.get("max_model_len", 32768)),
        enforce_eager=as_bool(cfg.get("enforce_eager", True)),
        gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.5)),
    )
    tokenizer = model.get_tokenizer()
    sampling = SamplingParams(
        temperature=float(cfg.get("temperature", 0.000001)),
        top_p=float(cfg.get("top_p", 0.9)),
        top_k=int(cfg.get("top_k", 1)),
        max_tokens=int(cfg.get("max_tokens", 2048)),
    )
    batch_size = int(cfg.get("batch_size", 64))
    expected_rollouts = int(cfg.get("expected_rollouts", 4))

    with output_path.open("a", encoding="utf-8") as out_f, failed_path.open("a", encoding="utf-8") as fail_f:
        for item in tqdm(todo, desc=f"verify split {index}", dynamic_ncols=True):
            try:
                meta_tasks, candidates = build_tasks(item, cfg)
                prompts: list[str] = []
                prompt_task_indices: list[int] = []
                for task_idx, (task, candidate) in enumerate(zip(meta_tasks, candidates)):
                    if not task["format_valid"]:
                        continue
                    prompts.append(format_prompt(tokenizer, str(item.get("question", "")), str(item.get("answer", "")), candidate))
                    prompt_task_indices.append(task_idx)

                verifier_texts_by_task: dict[int, str] = {}
                for start in range(0, len(prompts), batch_size):
                    outputs = model.generate(prompts[start : start + batch_size], sampling)
                    for offset, output in enumerate(outputs):
                        task_idx = prompt_task_indices[start + offset]
                        verifier_texts_by_task[task_idx] = output.outputs[0].text

                judgments = []
                for task_idx, task in enumerate(meta_tasks):
                    if not task["format_valid"]:
                        label = "B"
                        judgment = "wrong"
                        verifier_response = f"SKIPPED_FORMAT_INVALID: {task['format_error']}"
                        parse_failed = False
                    else:
                        verifier_response = verifier_texts_by_task.get(task_idx, "")
                        label = process_judgment(verifier_response)
                        parse_failed = not bool(label)
                        if parse_failed:
                            label = "B"
                            judgment = "wrong"
                        else:
                            judgment = {"A": "correct", "B": "wrong", "C": "invalid"}[label]
                    judgments.append(
                        {
                            **task,
                            "verifier_response": verifier_response,
                            "label": label,
                            "judgment": judgment,
                            "_parse_failed": parse_failed,
                        }
                    )
                correct = sum(1 for j in judgments if j["label"] == "A")
                wrong = sum(1 for j in judgments if j["label"] == "B")
                invalid = sum(1 for j in judgments if j["label"] == "C")
                parse_failed = sum(1 for j in judgments if j.pop("_parse_failed", False))
                verify_valid = len(judgments) == expected_rollouts and parse_failed < expected_rollouts
                result = {
                    "index": item.get("index"),
                    "id": item.get("id"),
                    "question": item.get("question"),
                    "answer": item.get("answer"),
                    "source": item.get("source"),
                    "original_id": item.get("original_id"),
                    "domain": item.get("domain"),
                    "type": item.get("type"),
                    "subtype": item.get("subtype"),
                    "verify_valid": verify_valid,
                    "num_rollouts": len(judgments),
                    "correct_count": correct,
                    "wrong_count": wrong,
                    "invalid_count": invalid,
                    "parse_failed_count": parse_failed,
                    "judgments": judgments,
                }
                dump_jsonl_line(out_f, result)
                if not result["verify_valid"]:
                    dump_jsonl_line(fail_f, result)
            except Exception as exc:
                err = {
                    "index": item.get("index"),
                    "id": item.get("id"),
                    "verify_valid": False,
                    "error": repr(exc),
                }
                dump_jsonl_line(fail_f, err)
                print(f"[ERROR] item {record_key(item)} failed: {exc}", file=sys.stderr, flush=True)

    final_rows = load_jsonl(output_path)
    valid_keys = {record_key(row) for row in final_rows if row.get("verify_valid")}
    valid_rows = len(valid_keys)
    if valid_rows >= len(all_items):
        stop_path.write_text(f"done {time.strftime('%F %T')}\n", encoding="utf-8")
        print(f"split={index} complete valid={valid_rows}/{len(all_items)}", flush=True)
    else:
        print(f"split={index} incomplete output_rows={len(final_rows)} valid={valid_rows}/{len(all_items)}", flush=True)


if __name__ == "__main__":
    main()
