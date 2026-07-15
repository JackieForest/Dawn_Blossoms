import json
import os
import argparse
import pandas as pd
from vlmeval.dataset import build_dataset
from openai import OpenAI
import asyncio
from openai import AsyncOpenAI
import json
from typing import Any, Dict, List, Tuple, Optional
import os
import base64
import io
import argparse
import pandas as pd
from tqdm import tqdm
import math
from PIL import Image
import time
import numpy as np

dataset_dir = "/share/wulijun/liyu/LMUData/oda"

TEMPERATURE = float(os.environ.get("EVAL_TEMPERATURE", "0.0"))
TOP_P = float(os.environ.get("EVAL_TOP_P", "0.95"))
MAX_TOKENS = int(os.environ.get("EVAL_MAX_TOKENS", "4096"))
REPETITION_PENALTY = float(os.environ.get("EVAL_REPETITION_PENALTY", "1.05"))
EVAL_ENABLE_THINKING = os.environ.get("EVAL_ENABLE_THINKING", "0").lower() in {"1", "true", "yes", "on"}
EVAL_REQUEST_TIMEOUT = int(os.environ.get("EVAL_REQUEST_TIMEOUT", "1800"))
IMAGE_TARGET_SIZES = [
    int(x) for x in os.environ.get("EVAL_IMAGE_TARGET_SIZES", "-1,1024,768,512").split(",")
    if x.strip()
]
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
THINKING_SYSTEM_PROMPT = (
    "You are a helpful assistant. Provide your analysis first. Then give the final answer "
    "enclosed in <answer>...</answer> tags."
)
SFT_SYSTEM_PROMPT = (
    "You are a helpful assistant. When answering the question, first provide the reasoning process enclosed in "
    "<think>...</think> tags. Then provide the final answer enclosed in <answer>...</answer> tags."
)

def proxy_off():
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = ''
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''

proxy_off()


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def resize_image_by_factor(img, factor=1):
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    img = img.resize((new_w, new_h))
    return img

def encode_image_to_base64(img, target_size=-1, fmt='JPEG'):
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    if target_size > 0:
        img = img.copy()
        img.thumbnail((target_size, target_size))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    image_data = img_buffer.getvalue()
    ret = base64.b64encode(image_data).decode('utf-8')
    max_size = 4194304
    min_edge = 0

    if min(img.size) < min_edge:
        factor = min_edge / min(img.size)
        image_new = resize_image_by_factor(img, factor)
        img_buffer = io.BytesIO()
        image_new.save(img_buffer, format=fmt)
        image_data = img_buffer.getvalue()
        ret = base64.b64encode(image_data).decode('utf-8')

    factor = 1
    while len(ret) > max_size:
        factor *= 0.7  # Half Pixels Per Resize, approximately
        image_new = resize_image_by_factor(img, factor)
        img_buffer = io.BytesIO()
        image_new.save(img_buffer, format=fmt)
        image_data = img_buffer.getvalue()
        ret = base64.b64encode(image_data).decode('utf-8')

    if factor < 1:
        new_w, new_h = image_new.size
        print(
            f'Warning: image size is too large and exceeds `VLMEVAL_MAX_IMAGE_SIZE` {max_size}, '
            f'resize to {factor:.2f} of original size: ({new_w}, {new_h})'
        )

    return ret

def is_context_length_error(exc):
    text = str(exc)
    return "max_tokens must be at least 1" in text or "maximum context length" in text


def is_retryable_generation_error(exc):
    text = str(exc).lower()
    return (
        is_context_length_error(exc)
        or "timed out" in text
        or "timeout" in text
        or "connection" in text
        or "server disconnected" in text
        or "internal server error" in text
    )


def get_system_prompt():
    custom_prompt = os.environ.get("EVAL_SYSTEM_PROMPT")
    if custom_prompt:
        return custom_prompt

    prompt_mode = os.environ.get("EVAL_SYSTEM_PROMPT_MODE", "").strip().lower()
    if prompt_mode in {"sft", "train", "training"}:
        return SFT_SYSTEM_PROMPT
    if prompt_mode in {"thinking", "think"}:
        return THINKING_SYSTEM_PROMPT
    if EVAL_ENABLE_THINKING:
        return THINKING_SYSTEM_PROMPT

    return DEFAULT_SYSTEM_PROMPT

def build_messages(item, image_target_size):
    instruct_prompt = get_system_prompt()
    messages = [
        {
            "role": "system",
            "content": instruct_prompt
        }
    ]

    user_content = []
    for msg in item['messages']:
        if msg['type'] == 'text':
            user_content.append({
                "type": "text",
                "text": msg['value']
            })
        elif msg['type'] == 'image':
            img = Image.open(msg['value'])
            b64 = encode_image_to_base64(img, target_size=image_target_size)
            img_struct = f'data:image/jpeg;base64,{b64}'
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": img_struct
                }
            })

    messages.append({
        "role": "user",
        "content": user_content
    })
    return messages

async def generate_async(model_name, item, client):
    index = item.get('index', 'unknown')
    
    for attempt, image_target_size in enumerate(IMAGE_TARGET_SIZES):
        start_time = time.time()
        try:
            if attempt > 0:
                print(f"\nRetrying item {index} with image_target_size={image_target_size}")
            messages = build_messages(item, image_target_size)
            
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                timeout=EVAL_REQUEST_TIMEOUT,
                extra_body={
                    "repetition_penalty": REPETITION_PENALTY,
                    "chat_template_kwargs": {"enable_thinking": EVAL_ENABLE_THINKING},
                }
            )

            result = response.choices[0].message.content
            return item, result
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"Item {index}: {e} after {elapsed:.2f}s - (image_target_size={image_target_size})")

            if not is_retryable_generation_error(e) or attempt == len(IMAGE_TARGET_SIZES) - 1:
                print(f"Item {index}: Max retries reached")
                return item, None
    
    return item, None

async def process_all_data(data, url, model_name, output_file, max_concurrent=320):
    host, port = url.rsplit(":", 1) if ":" in url else (url, "8000")
    client = AsyncOpenAI(
        base_url=f"http://{host}:{port}/v1",
        api_key="EMPTY"
    )
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate_with_limit(item):
        async with semaphore:
            return await generate_async(model_name, item, client)
    
    tasks = []
    for item in data:
        task = asyncio.create_task(generate_with_limit(item))
        tasks.append(task)
    
    print(f"Created {len(tasks)} tasks, all submitted to server...")
    
    pbar = tqdm(total=len(tasks), desc="Processing")
    
    success_count = 0
    fail_count = 0
    
    with open(output_file, 'a', encoding='utf-8') as f:
        for completed_task in asyncio.as_completed(tasks):
            try:
                item, result = await completed_task
                
                if result:
                    item['prediction'] = result
                    item.pop('messages', None)
                    f.write(json.dumps(make_json_serializable(item), ensure_ascii=False) + '\n')
                    f.flush()
                    success_count += 1
                else:
                    fail_count += 1
                    print(f"\nFailed to process item {item.get('index', 'unknown')}")
                    
                pbar.update(1)
                pbar.set_postfix({'success': success_count, 'failed': fail_count})
            
                
            except Exception as e:
                print(f"\nError processing completed task: {e}")
                fail_count += 1
                pbar.update(1)
                pbar.set_postfix({'success': success_count, 'failed': fail_count})
                continue
    
    pbar.close()
    print(f"\nProcessing complete! Success: {success_count}, Failed: {fail_count}")

def load_existing_indexes(path):
    if not os.path.exists(path):
        return set()
    idxs = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                idxs.add(str(obj['index'])+str(obj['dataset'])+str(obj['split']))
            except Exception:
                continue
    return idxs
 
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--url", type=str)
    parser.add_argument("--resume", type=bool, default=True)
    parser.add_argument("--max_concurrent", type=int, default=250)
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    data = []
    for dataset_name in datasets:
        dataset = build_dataset(dataset_name)
        print(f"Loading dataset: {dataset_name} with {len(dataset)} records")
        for i in range(len(dataset)):            
            messages = dataset.build_prompt(i)
            for message in messages:
                if message['type'] == 'text':
                    # for CharXiv
                    message['value'] = message['value'].replace("* Your final answer must be grounded to some text that is explicitly written and relevant to the question in the chart.\n    * If you need to answer multiple terms, separate them with commas.\n    * Unless specified in the question (such as answering with a letter), you are required to answer the full names of subplots and/or labels by default.", "").replace("* If there are options in the question, your final answer must conform to one of the options.\n    * If there are additional instructions in the question, follow them accordingly.\n    * If there are neither options nor additional instructions, you are allowed to respond with a short phrase only","").replace("* Your final answer must be grounded to a number that is exlicitly written and relevant to the question in the chart, even if it's an approximate value.\n    * You are allowed to extract numbers within some text when needed.","").replace("* Your final answer must be an exact integer.\n", "").strip()
                    # for MathVision
                    message['value'] = message['value'].replace("Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: ","").replace("Hint: Please answer the question and provide the final answer at the end.\nQuestion: ", "").strip()
                    question = message['value']
            data.append({
                "dataset": dataset_name,
                "index": i,
                "messages": messages,
                "question": question,
                "answer": dataset[i]['answer'],
                "split": dataset[i]['split'] if 'split' in dataset[i] else 'default'
            })

            # question = ""
            # for message in messages:
            #     if message['type'] == 'text':
            #         question = message['value']
                 
            # data.append({
            #     "dataset": dataset_name,
            #     "index": i,
            #     "messages": messages,
            #     "question": question,
            #     "answer": dataset[i]['answer'],
            #     "split": dataset[i]['split'] if 'split' in dataset[i] else 'default'
            # })
    
    model = os.environ.get("EVAL_MODEL_ALIAS") or os.path.basename(args.model_name)
    output_root = os.environ.get("EVAL_OUTPUT_ROOT", ".")
    output_path = os.path.join(output_root, "outputs", model, "rollout.jsonl")
    print(f"Output path: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Loaded {len(data)} records")
    
    if args.resume:
        existing_ids = load_existing_indexes(output_path)
        print(f"Found {len(existing_ids)} already processed records")

        data = [item for item in data if str(item['index'])+str(item['dataset'])+str(item['split']) not in existing_ids]
        print(f"Remaining {len(data)} records to process")
    else:
        os.remove(output_path) if os.path.exists(output_path) else None
        
    await process_all_data(data, args.url, args.model_name, output_path, max_concurrent=args.max_concurrent)
    
if __name__ == "__main__":
    asyncio.run(main())

    
        
        
