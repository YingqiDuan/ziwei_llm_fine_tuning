"""
Extract prompt/completion pairs from a JSONL answer dataset.

Usage:
    python extract_prompt_completion.py --input dataset/ziwei_answer_dataset.jsonl --output dataset/ziwei_prompt_completion.jsonl
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter prompt/completion fields from a JSONL dataset."
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
        help="Optional path for the filtered dataset (JSONL). Defaults to <input>_prompt_completion.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.input.with_name(
        f"{args.input.stem}_prompt_completion{args.input.suffix}"
    )

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
            json.dump({"prompt": prompt, "completion": completion}, dst, ensure_ascii=False)
            dst.write("\n")


if __name__ == "__main__":
    main()

