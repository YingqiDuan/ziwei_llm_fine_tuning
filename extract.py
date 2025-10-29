"""
Convert a JSONL answer dataset directly into Ollama chat fine-tuning records.

Example:
    python extract.py --input dataset/ziwei_answer_dataset.jsonl \
        --prefer-record-system-prompt
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform prompt/completion entries to Ollama chat conversations."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the input answer dataset (JSONL).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output JSONL path. Defaults to <input>_ollama.jsonl.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Override system prompt to inject into each Ollama record.",
    )
    parser.add_argument(
        "--prefer-record-system-prompt",
        action="store_true",
        help="Use each input record's system_prompt field when available.",
    )
    return parser.parse_args()


def derive_default_output_path(input_path: Path) -> Path:
    suffix = input_path.suffix or ".jsonl"
    stem = input_path.stem
    return input_path.with_name(f"{stem}_ollama{suffix}")


def main() -> None:
    args = parse_args()
    output_path = args.output or derive_default_output_path(args.input)

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
                raise ValueError(
                    f"Missing prompt/completion on line {lineno} of {args.input}"
                )

            system_prompt = args.system_prompt
            if system_prompt is None and args.prefer_record_system_prompt:
                system_prompt = record.get("system_prompt")

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
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

