"""
Run inference with a Unsloth FastLanguageModel + fine-tuned LoRA adapter.

用法示例：

python infer_unsloth.py \
  --model-path outputs/gpt-oss-20b-qlora-test \
  --prompt-file prompt.txt
"""

import argparse
import sys
from pathlib import Path
import threading

import unsloth
from unsloth import FastLanguageModel
import torch
from transformers import TextIteratorStreamer

from prompts import DEFAULT_SYSTEM_PROMPT  # 和你原来的复用


def parse_args():
    parser = argparse.ArgumentParser("Run a quick completion with Unsloth + LoRA adapter.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="LoRA adapter 目录（就是 train_qlora_unsloth.py 的 output_dir，或者上传到 HF 后的 repo 名）。",
    )
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
        help="System prompt prepended to the user prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation length.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling.")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable incremental printing; wait for the full completion instead.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Max sequence length to load with Unsloth.",
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

    max_seq_length = args.max_seq_length
    dtype = None          # 让 Unsloth 自己选（通常是 bf16）
    load_in_4bit = True   # 你的权重本来就是 unsloth bnb-4bit + LoRA

    # 关键：这里直接用 LoRA adapter 目录来 load
    # Unsloth 会自动从 adapter 里找到 base model 名并加载到 GPU 上
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # 开启 Unsloth 的推理优化（2x faster）
    FastLanguageModel.for_inference(model)
    torch.cuda.empty_cache()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompt = build_prompt(args.system_prompt, user_prompt)

    # 把输入直接丢到 CUDA 上（model 已经在显存里）
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to("cuda")

    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask", None),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": True,
        "use_cache": True,
    }

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
