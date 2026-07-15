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


CV_COT_PROMPT = """As a grading expert, your task is to determine whether the candidate's response correctly solves the problem according to the provided standard answer. Follow these evaluation guidelines precisely:

Evaluation Protocol:
1. Reference Standard:
   - The standard answer is definitive and always correct.
   - The question is perfectly valid - never question it.
   - Do not regenerate answers; only compare the candidate response with the given standard answer.

2. Comparison Method:
   - Carefully analyze the question's requirements and the standard answer's structure.
     * Determine whether the question expects exact matching of the entire standard answer or allows semantic or partial matching of its components.
     * Determine whether the question requires only a final answer, or also requires setup, derivation, equation, proof, explanation, multiple parts, structured output, or action format.
     * This determination must be made based on the question's phrasing and the nature of the standard answer.

   - Compare primarily the candidate's final answer, and use the reasoning block as auxiliary evidence for judging correctness when necessary.

   - If the final answer is concise but the reasoning contains required setup, derivation, equation, proof, intermediate result, or explanation explicitly requested by the question, judge the full response as correct.

   - If the final answer is concise or refers to the reasoning, such as "see proof", "as shown above", or "therefore true", and the required proof/setup/derivation is correctly present in the reasoning, judge the full response as correct.

   - For short descriptive answers, judge semantic equivalence rather than exact wording. If the candidate preserves the core meaning of the standard answer, do not mark it incorrect only because of wording, granularity, or presentation differences.
     * For example, if the standard answer describes a slight increase/enhancement and the candidate describes the same increasing/enhancing trend in context, it may be considered correct.
     * If the standard answer uses a time description and the candidate gives an equivalent or contextually matching time interval, it may be considered correct.
     * However, if the candidate changes the direction, relation, object, interval, quantity, option, or required conclusion, judge it as incorrect.
     * If the standard answer contains a specific number, time span, interval, option, object, coordinate, or named entity, a vague or generic candidate answer is not correct unless it clearly identifies the same required value or entity in context.

   - Ignore reasoning/explanation errors that do not affect the final answer or task completion.
   - Disregard any non-critical differences in formatting, wording, ordering, or presentation style unless the question explicitly requires them.
   - For mathematical expressions: calculate step by step whether the formulas or results are equivalent.
   - For numerical answers: allow reasonable rounding, equivalent units, or equivalent numeric forms when appropriate.
   - For multiple-choice questions: compare the final selected choice and corresponding option content.
   - For structured output tasks such as JSON, bbox, point_2d, coordinate, GUI action, or action dictionary, judge strictly based on the final answer format and content. Do not repair an invalid structured final answer using the reasoning.

3. Multi-part Answers:
   - For questions requiring multiple responses, all required parts must be present either in the final answer or explicitly in the reasoning.
   - If a required part is missing from both the final answer and the reasoning, judge the response as incorrect.
   - For structured multi-part answers, all required fields or components must appear in the final answer. Partial matches are considered incorrect.
   - For non-structured multi-part reasoning answers, judge as correct if all required components are clearly present across the final answer and reasoning.

4. Validity Check:
   - Reject responses that are:
     * Incomplete (cut off mid-sentence, missing the final answer, or lacking a complete response) -> Label as INCOMPLETE.
     * Repetitive (repetition of words or phrases in a clear loop) -> Label as REPETITIVE.
     * Explicit refusals (e.g., directly return "I cannot answer/provide/access ...") -> Label as REFUSAL.
   - Do not mark a response as incomplete or repetitive solely because the reasoning is long.
   - For invalid responses, specify the type in the judgment (e.g., \\boxed{{C}} - INCOMPLETE).

Grading Scale:
\\boxed{{A}} - CORRECT:
   - The candidate response correctly solves the problem.
   - The final answer matches the standard answer exactly or is semantically/equivalently correct.
   - For numerical answers: values are equivalent under reasonable rounding or tolerance.
   - For mathematical expressions: equivalent expressions, equations, or solved results are correct.
   - For short descriptive answers: semantically equivalent wording or contextually equivalent granularity is correct.
   - For questions requiring setup, derivation, equation, proof, or explanation, those required components may appear in the reasoning even if the final answer is concise.

\\boxed{{B}} - INCORRECT:
   - The candidate response gives a materially different answer from the standard answer.
   - The final answer is wrong, even if the reasoning contains some correct statements.
   - A required component is missing from both the final answer and the reasoning.
   - The candidate changes the required object, direction, relation, quantity, interval, option, or conclusion.
   - For structured output tasks, the final answer has invalid schema, wrong fields, wrong coordinates, wrong action, wrong selected option, or missing required components.

\\boxed{{C}} - INCOMPLETE/REPETITIVE/REFUSAL:
   - Fails validity criteria above (must specify: INCOMPLETE/REPETITIVE/REFUSAL).

Execution Steps and Output Formats:

Analysis step by step: [
Thoroughly evaluate the candidate response including:
(1) First check if the response is INCOMPLETE, REPETITIVE, or a REFUSAL. If so, immediately classify as \\boxed{{C}} with the corresponding type.
(2) Analyze the question's core requirements and the standard answer's structure, for example:
- Strict requirements: mandatory constraints such as exact option, schema, coordinates, answer order, multi-part completeness, proof, equation, or derivation.
- Tolerant allowances: semantic equivalence, equivalent expressions, equivalent wording, reasonable rounding, missing non-critical formatting, or contextually equivalent granularity.
- Required answer type, precision level, setup/equation/derivation/proof/explanation requirements, etc.
(3) Compare the candidate response with the standard answer:
- Primarily compare the final answer.
- Use the reasoning block as auxiliary evidence when the question explicitly requires setup, equation, derivation, proof, intermediate result, or explanation.
- For short descriptive answers, judge whether the core semantic meaning is preserved.
- For structured output tasks, judge the final answer strictly.
- Decide whether the candidate response correctly solves the problem.
]
Final Judgment: \\boxed{{A/B/C}} - <CORRECT/INCORRECT/INCOMPLETE/REPETITIVE/REFUSAL>

Here is your task.
<Original Question Begin>
{question}
<Original Question End>

<Standard Answer Begin>
{gold_answer}
<Standard Answer End>

<Candidate Response Begin>
{llm_response}
<Candidate Response End>

Analysis step by step and Final Judgment:
"""

SYSTEM_PROMPT = "Please act as a grading expert and judge whether the candidate response correctly solves the problem according to the standard answer."


def truncate(value: Any, max_chars: int = 50000, preserve_final_answer: bool = False) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars] + "\n[TRUNCATED]"
    if preserve_final_answer:
        matches = list(re.finditer(r"<answer>.*?</answer>", text, flags=re.DOTALL | re.IGNORECASE))
        if matches:
            final_answer = matches[-1].group(0)
            if not text[matches[-1].end() :].strip() and final_answer not in truncated:
                truncated += "\n" + final_answer
    return truncated


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
    if not re.search(r"<answer>.*?</answer>\s*$", text, flags=re.DOTALL | re.IGNORECASE):
        return False, "bad_final_line"
    return True, ""


def format_prompt(tokenizer: AutoTokenizer, question: str, answer: str, candidate: str) -> str:
    prompt = CV_COT_PROMPT.format(
        question=truncate(question),
        gold_answer=truncate(answer),
        llm_response=truncate(candidate, preserve_final_answer=True),
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
    parser = argparse.ArgumentParser(description="Verify 27B distill rollouts with CompassVerifier.")
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
