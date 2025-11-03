"""
Run inference with a base model plus saved LoRA adapter.
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser("Run a quick chat completion with the fine-tuned adapter.")
    parser.add_argument("--base-model", required=True, help="Base Hugging Face model id or path.")
    parser.add_argument("--adapter", required=True, help="Directory containing the saved LoRA adapter.")
    parser.add_argument("--prompt", required=True, help="User message to send to the model.")
    parser.add_argument(
        "--system-prompt",
        default="你是紫微斗数顾问，请结合命盘信息给出建议。",
        help="Optional system prompt prepended to the conversation.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation length.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling.")
    return parser.parse_args()


def build_chat_prompt(system_prompt: str, user_prompt: str) -> str:
    if system_prompt:
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    return f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"


def main() -> None:
    args = parse_args()

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

    prompt = build_chat_prompt(args.system_prompt, args.prompt)
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
