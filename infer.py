"""
Run inference with a base model plus the fine-tuned LoRA adapter.
Supports inline, file-based, or interactive prompts.
"""

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import DEFAULT_SYSTEM_PROMPT


def parse_args():
    parser = argparse.ArgumentParser("Run a quick chat completion with the fine-tuned adapter.")
    parser.add_argument("--base-model", required=True, help="Base Hugging Face model id or path.")
    parser.add_argument("--adapter", required=True, help="Directory containing the saved LoRA adapter.")
    parser.add_argument(
        "--prompt",
        help="Inline user prompt. Overrides --prompt-file and interactive input if provided.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Read the user prompt from a text file (used when --prompt is omitted).",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt prepended to the conversation (defaults to shared Grok prompt).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation length.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling.")
    return parser.parse_args()


def build_chat_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def resolve_user_prompt(prompt: str | None, prompt_file: Path | None) -> str:
    if prompt is not None:
        return prompt
    if prompt_file:
        return prompt_file.read_text(encoding="utf-8").strip()

    print("Enter user prompt (finish with an empty line):", file=sys.stderr)
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line:
            break
        lines.append(line)

    text = "\n".join(lines).strip()
    if not text:
        raise ValueError("Prompt is empty; provide --prompt, --prompt-file, or enter text interactively.")
    return text


def main() -> None:
    args = parse_args()
    user_prompt = resolve_user_prompt(args.prompt, args.prompt_file)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prompt = build_chat_prompt(args.system_prompt, user_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(completion.strip())


if __name__ == "__main__":
    main()
