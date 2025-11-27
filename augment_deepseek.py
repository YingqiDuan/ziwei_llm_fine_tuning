"""
Dataset augmentation script using DeepSeek chat completion API.

- Reads records with `chart_text` from an input JSONL file
- Calls DeepSeek chat API to get completions
- Writes enriched records to an output JSONL file
"""

import argparse
import json
import os
import time
from typing import Dict, Iterable, List, Set

from dotenv import load_dotenv
from openai import OpenAI

from prompts import DEFAULT_SYSTEM_PROMPT  # 与原脚本保持一致


DEFAULT_MODEL = "deepseek-chat"
DEFAULT_SLEEP = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Call LLM to augment chart_text records (DeepSeek).")
    parser.add_argument("--input", required=True, help="Input JSONL with chart_text field.")
    parser.add_argument("--output", required=True, help="Destination JSONL for completions.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DeepSeek model name.")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt (empty string disables system message).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum records to generate.")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="Pause between calls.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing output and skip records whose `prompt` already exists.",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com",
        help="DeepSeek API base URL.",
    )
    return parser.parse_args()


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("请先在环境变量中设置 DEEPSEEK_API_KEY。")
    return api_key


def load_existing_prompts(path: str) -> Set[str]:
    """从已有的输出文件里收集已生成过的 prompt，用于 --resume 跳过。"""
    prompts: Set[str] = set()
    if not os.path.exists(path):
        return prompts

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = data.get("prompt")
            if prompt:
                prompts.add(prompt)
    return prompts


def build_messages(text: str, system_prompt: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})
    return messages


def call_deepseek(
    client: OpenAI,
    model: str,
    text: str,
    system_prompt: str,
) -> str:
    """调用 DeepSeek Chat Completion，返回字符串内容。"""
    resp = client.chat.completions.create(
        model=model,
        messages=build_messages(text, system_prompt),
        max_tokens=4096,
        temperature=0.7,
        top_p=1.0,
        stream=False,
    )
    if not resp.choices:
        raise RuntimeError(f"响应缺少 choices: {resp}")
    return resp.choices[0].message.content.strip()


def iterate_records(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    args = parse_args()
    api_key = load_api_key()

    # 统一创建 client，避免每次请求都新建
    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
    )

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # 处理 resume 逻辑
    existing_prompts = load_existing_prompts(args.output) if args.resume else set()
    mode = "a" if args.resume and os.path.exists(args.output) else "w"

    generated = 0
    with open(args.output, mode, encoding="utf-8") as out_handle:
        for record in iterate_records(args.input):
            chart_text = record.get("chart_text")
            if not chart_text:
                continue

            if chart_text in existing_prompts:
                # 已经生成过，跳过
                continue

            completion = call_deepseek(
                client=client,
                model=args.model,
                text=chart_text,
                system_prompt=args.system_prompt,
            )

            enriched = {
                **record,
                "model": args.model,
                "system_prompt": args.system_prompt,
                "prompt": chart_text,
                "completion": completion,
            }

            out_handle.write(json.dumps(enriched, ensure_ascii=False))
            out_handle.write("\n")
            out_handle.flush()

            existing_prompts.add(chart_text)
            generated += 1

            if args.limit is not None and generated >= args.limit:
                break

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"[augment] Generated {generated} completions -> {args.output}")


if __name__ == "__main__":
    main()
