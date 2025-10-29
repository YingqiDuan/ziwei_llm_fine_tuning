"""
Convert prompt/completion JSONL data to the chat-style JSONL format expected by Ollama fine-tuning.

Usage:
    python convert_prompt_completion_to_ollama.py \\
        --input dataset/ziwei_prompt_completion.jsonl \\
        --output dataset/ziwei_ollama_train.jsonl
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform prompt/completion JSONL entries into Ollama chat fine-tuning format."
    )
    parser.add_argument("--input", required=True, type=Path, help="Source JSONL file.")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSONL file in Ollama chat format.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system prompt to prepend as the first message in each conversation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.input.open("r", encoding="utf-8") as src, args.output.open(
        "w", encoding="utf-8"
    ) as dst:
        for lineno, line in enumerate(src, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record.get("prompt")
            completion = record.get("completion")
            if prompt is None or completion is None:
                raise ValueError(
                    f"Missing prompt/completion on line {lineno} of {args.input}"
                )

            messages = []
            if args.system_prompt:
                messages.append({"role": "system", "content": args.system_prompt})
            messages.extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
            )

            json.dump({"messages": messages}, dst, ensure_ascii=False)
            dst.write("\n")


if __name__ == "__main__":
    main()

