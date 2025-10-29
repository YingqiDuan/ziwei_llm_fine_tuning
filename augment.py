from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from dotenv import load_dotenv


import certifi


API_URL = "https://api.x.ai/v1/chat/completions"
DEFAULT_MODEL = "grok-4-fast"
DEFAULT_SYSTEM_PROMPT = (
    "你现在是资深的国学易经术数领域专家，请详细分析下面这个紫微斗数命盘，综合使用三合紫微、飞星紫微、钦天四化等各流派紫微斗数的分析技法，对命盘十二宫星曜分布和各宫位间的飞宫四化进行细致分析，进而对命主的健康、学业、事业、财运、人际关系、婚姻和感情等各个方面进行全面分析和总结。"
)
DEFAULT_SLEEP = 0.5  
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


@dataclass(frozen=True)
class Record:
    raw: dict
    prompt: str
    completion: str
    is_existing: bool = False


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/ziwei_chart_dataset.jsonl"),
        help="Source dataset containing chart_text (default: dataset/ziwei_chart_dataset.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/ziwei_answer_dataset.jsonl"),
        help="Destination dataset with Grok completions (default: dataset/ziwei_answer_dataset.jsonl).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Grok model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt content; empty string disables system message.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Optional limit on number of records to process.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="max_tokens parameter for Grok call (default: 4096).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="temperature parameter for Grok call (default: 0.7).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="top_p parameter for Grok call (default: 1.0).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Maximum retry attempts per request (default: 5).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP,
        help=f"Seconds to sleep between successful requests (default: {DEFAULT_SLEEP}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file (skip already processed prompts).",
    )
    return parser.parse_args(argv)


def _load_existing(output_path: Path) -> dict[str, dict]:
    if not output_path.exists():
        return {}
    existing: dict[str, dict] = {}
    with output_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            prompt = data.get("prompt")
            if prompt:
                existing[prompt] = data
    return existing


def _iter_input_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _build_messages(
    chart_text: str,
    *,
    system_prompt: str,
) -> tuple[list[dict[str, str]], str]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": chart_text})
    return messages, chart_text


def _call_grok(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    retries: int,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
    ).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Connection": "keep-alive",
        "Authorization": f"Bearer {api_key}",
    }

    last_error: Optional[Exception] = None
    backoff = 1.0
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            API_URL, data=payload, headers=headers, method="POST"
        )
        retryable = True
        jitter = False
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CONTEXT) as resp:
                content = resp.read()
            data = json.loads(content)
            choices = data.get("choices")
            if not choices:
                raise ValueError(f"响应中没有 choices 字段: {data}")
            return choices[0]["message"]["content"].strip()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "ignore")
            cf_ray = exc.headers.get("cf-ray") if exc.headers else None
            server = exc.headers.get("Server") if exc.headers else None
            meta = []
            if cf_ray:
                meta.append(f"cf-ray={cf_ray}")
            if server:
                meta.append(f"server={server}")
            meta_suffix = f" [{', '.join(meta)}]" if meta else ""
            message = (
                f"HTTP {exc.code}: {body or exc.reason}{meta_suffix} "
                f"(attempt {attempt}/{retries})"
            )
            if exc.code in {401, 403}:
                raise RuntimeError(message + " (non-retriable)") from exc
            retryable = exc.code in {408, 425, 429} or 500 <= exc.code < 600
            jitter = exc.code in {429} or 500 <= exc.code < 600
            last_error = RuntimeError(message)
        except urllib.error.URLError as exc:
            last_error = RuntimeError(
                f"网络错误: {exc.reason} (attempt {attempt}/{retries})"
            )
            jitter = True
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            retryable = False
        if not retryable:
            break

        if attempt < retries:
            sleep_time = backoff
            if jitter:
                sleep_time *= random.uniform(0.5, 1.5)
            time.sleep(sleep_time)
            backoff *= 2
    assert last_error is not None
    raise last_error


def _stream_records_with_grok(
    *,
    inputs: Iterable[dict],
    api_key: str,
    model: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    retries: int,
    sleep_seconds: float,
    resume_map: dict[str, dict],
    limit: Optional[int],
) -> Iterable[Record]:
    generated = 0
    for data in inputs:
        chart_text = data.get("chart_text")
        if not chart_text:
            continue

        messages, prompt = _build_messages(chart_text, system_prompt=system_prompt)

        if resume_map and prompt in resume_map:
            completion = resume_map[prompt]["completion"]
            yield Record(raw=data, prompt=prompt, completion=completion, is_existing=True)
            continue

        completion = _call_grok(
            api_key=api_key,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            retries=retries,
        )
        yield Record(raw=data, prompt=prompt, completion=completion)
        generated += 1
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        if limit is not None and generated >= limit:
            break


def main(argv: Optional[list[str]] = None) -> None:
    load_dotenv()

    args = _parse_args(argv)

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise EnvironmentError("请先通过环境变量 XAI_API_KEY 提供 xAI API Key。")

    if not args.input.exists():
        raise FileNotFoundError(f"未找到输入文件: {args.input}")

    resume_map = _load_existing(args.output) if args.resume else {}

    inputs = _iter_input_records(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.resume and resume_map else "w"
    with args.output.open(mode, encoding="utf-8") as fh:
        for record in _stream_records_with_grok(
            inputs=inputs,
            api_key=api_key,
            model=args.model,
            system_prompt=args.system_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            retries=args.retries,
            sleep_seconds=args.sleep,
            resume_map=resume_map,
            limit=args.limit,
        ):
            if record.is_existing:
                continue
            enriched = {
                **record.raw,
                "model": args.model,
                "system_prompt": args.system_prompt,
                "prompt": record.prompt,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "completion": record.completion,
            }
            fh.write(json.dumps(enriched, ensure_ascii=False))
            fh.write("\n")
            fh.flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("已终止 (Ctrl+C)")
