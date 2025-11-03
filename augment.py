"""
Minimal dataset augmentation script for calling the xAI Grok chat completion API.
Reads `chart_text` records from an input JSONL file and writes completions to an output JSONL file.
"""

import argparse
import json
import os
import ssl
import time
import urllib.request

import certifi
from dotenv import load_dotenv

from prompts import DEFAULT_SYSTEM_PROMPT

API_URL = "https://api.x.ai/v1/chat/completions"
DEFAULT_MODEL = "grok-4-fast"
DEFAULT_SLEEP = 0.5
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def parse_args():
    parser = argparse.ArgumentParser("Call Grok to augment chart_text records.")
    parser.add_argument("--input", required=True, help="Input JSONL with chart_text field.")
    parser.add_argument("--output", required=True, help="Destination JSONL for completions.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Grok model name.")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt (empty string disables system message).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum records to generate.")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="Pause between calls.")
    parser.add_argument("--resume", action="store_true", help="Append to existing output and skip duplicates.")
    return parser.parse_args()


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 XAI_API_KEY 以调用 xAI API。")
    return api_key


def load_existing_prompts(path: str) -> set[str]:
    prompts: set[str] = set()
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


def build_messages(text: str, system_prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})
    return messages


def call_grok(api_key: str, model: str, text: str, system_prompt: str) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": build_messages(text, system_prompt),
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 1.0,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60, context=_SSL_CONTEXT) as resp:
        data = json.loads(resp.read())

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"响应缺少 choices: {data}")
    return choices[0]["message"]["content"].strip()


def iterate_records(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    args = parse_args()
    api_key = load_api_key()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    existing_prompts = load_existing_prompts(args.output) if args.resume else set()
    mode = "a" if args.resume and os.path.exists(args.output) else "w"

    generated = 0
    with open(args.output, mode, encoding="utf-8") as out_handle:
        for record in iterate_records(args.input):
            chart_text = record.get("chart_text")
            if not chart_text:
                continue
            if chart_text in existing_prompts:
                continue

            completion = call_grok(api_key, args.model, chart_text, args.system_prompt)
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
