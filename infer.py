"""
Run inference with a base model plus the fine-tuned LoRA adapter.
Supports inline, file-based, or interactive prompts.
"""

import argparse
import sys
import threading
from pathlib import Path

import transformers

# 🔧 关掉 Transformers 的 caching allocator warmup，避免一次性申请一大块显存
try:
    import transformers.modeling_utils as modeling_utils

    def _no_warmup(*args, **kwargs):
        return

    modeling_utils.caching_allocator_warmup = _no_warmup
except Exception as e:
    print("[infer] Warning: failed to disable caching_allocator_warmup:", e)

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

from prompts import DEFAULT_SYSTEM_PROMPT


def parse_args():
    parser = argparse.ArgumentParser("Run a quick completion with the fine-tuned adapter.")
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
        help="System prompt prepended to the user prompt (defaults to shared Grok prompt).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation length.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling.")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float16", "float32", "bfloat16"),
        help="Model dtype when not using 4-bit loading (auto = float16 on CUDA else float32).",
    )
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help="Load the base model with 4-bit quantization (requires bitsandbytes).",
    )
    parser.add_argument(
        "--device-map",
        default="cuda",
        help='Device map passed to from_pretrained (e.g. "auto", "cuda", "cpu").',
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable incremental printing; wait for the full completion instead.",
    )
    return parser.parse_args()


def build_prompt(system_prompt: str, user_prompt: str) -> str:
    parts = []
    if system_prompt:
        parts.append(system_prompt.strip())
    if user_prompt:
        parts.append(user_prompt.strip())
    return "\n\n".join(part for part in parts if part)


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

    quant_config = None
    if args.load_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    dtype_lookup: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    resolved_dtype = args.dtype
    if resolved_dtype == "auto":
        resolved_dtype = "float16" if torch.cuda.is_available() else "float32"
    torch_dtype = None if args.load_4bit else dtype_lookup[resolved_dtype]

    model_kwargs: dict[str, object] = {
        "device_map": args.device_map,
        "trust_remote_code": True,
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **model_kwargs,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prompt = build_prompt(args.system_prompt, user_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": True,
    }
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        generation_kwargs["attention_mask"] = attention_mask

    if args.no_stream:
        with torch.no_grad():
            output_ids = model.generate(**generation_kwargs)
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(completion.strip())
        return

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs["streamer"] = streamer

    worker = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    worker.start()
    try:
        for chunk in streamer:
            print(chunk, end="", flush=True)
    finally:
        worker.join()
        print()


if __name__ == "__main__":
    main()
