"""
Minimal converter from prompt/completion JSONL to Ollama chat format.
Prefers record-specific system prompts when present, but allows a CLI override.
"""

import argparse
import json
from pathlib import Path

from prompts import DEFAULT_SYSTEM_PROMPT


def parse_args():
    parser = argparse.ArgumentParser("Convert prompt/completion pairs to Ollama chat JSONL.")
    parser.add_argument("--input", required=True, type=Path, help="Source JSONL file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination JSONL. Defaults to <input>_ollama.jsonl.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="Override system prompt for every conversation (defaults to shared Grok prompt).",
    )
    return parser.parse_args()


def default_output_path(path: Path) -> Path:
    suffix = path.suffix or ".jsonl"
    return path.with_name(f"{path.stem}_ollama{suffix}")


def resolve_system_prompt(cli_prompt: str | None, record: dict) -> str | None:
    if cli_prompt is not None:
        return cli_prompt
    return record.get("system_prompt")


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path(args.input)

    with args.input.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for lineno, line in enumerate(src, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record.get("prompt")
            completion = record.get("completion")
            if prompt is None or completion is None:
                raise ValueError(f"Missing prompt/completion at line {lineno}")

            system_prompt = resolve_system_prompt(args.system_prompt, record)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": completion})

            json.dump({"messages": messages}, dst, ensure_ascii=False)
            dst.write("\n")

    print(f"[extract] Wrote Ollama chat dataset to {output_path}")


if __name__ == "__main__":
    main()
