"""
Convert prompt/completion JSONL into a flattened format for supervised tuning.
Supports optional system prompts by prepending them to the user prompt content.
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser("Normalize prompt/completion pairs for fine-tuning.")
    parser.add_argument("--input", required=True, type=Path, help="Source JSONL file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination JSONL. Defaults to <input>_prompt_completion.jsonl.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        help=(
            "Optional system prompt to prepend to each prompt. "
            "Record-level system prompts take precedence when present; "
            "no system prompt is added when omitted."
        ),
    )
    return parser.parse_args()


def default_output_path(path: Path) -> Path:
    suffix = path.suffix or ".jsonl"
    return path.with_name(f"{path.stem}_prompt_completion{suffix}")


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
            prompt_chunks = []
            if system_prompt:
                prompt_chunks.append(system_prompt.strip())
            prompt_chunks.append(prompt)
            merged_prompt = "\n\n".join(chunk for chunk in prompt_chunks if chunk)

            json.dump({"prompt": merged_prompt, "completion": completion}, dst, ensure_ascii=False)
            dst.write("\n")

    print(f"[extract] Wrote prompt/completion dataset to {output_path}")


if __name__ == "__main__":
    main()
