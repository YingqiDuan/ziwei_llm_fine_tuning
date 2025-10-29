"""
Supervised fine-tuning (SFT) entry point for gpt-oss-20B (or compatible chat models)
using QLoRA on a local Ollama-style dataset.

Example:
    python train_qlora.py \
        --model-name unsloth/gpt-oss-20b \
        --dataset-path dataset/ziwei_ollama_train.jsonl \
        --output-dir outputs/gpt-oss-20b-qlora
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


@dataclass
class ScriptArgs:
    model_name: str
    dataset_path: str
    output_dir: str
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: Optional[float]
    max_steps: int
    max_seq_length: int
    weight_decay: float
    warmup_ratio: float
    logging_steps: int
    save_steps: int
    eval_steps: Optional[int]
    evaluation_strategy: str
    seed: int
    validation_split: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: Optional[List[str]]
    bf16: bool
    packing: bool
    gradient_checkpointing: bool
    system_prompt_override: Optional[str]
    prefer_record_system_prompt: bool


def parse_args() -> ScriptArgs:
    parser = argparse.ArgumentParser(
        description="QLoRA SFT trainer for chat datasets in Ollama JSONL format."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Base model name or local path (e.g., unsloth/gpt-oss-20b or ./gpt-oss-20b).",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to the JSONL dataset in Ollama chat format.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store the trained LoRA adapter and tokenizer.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per device batch size for training.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=1,
        help="Per device batch size for evaluation.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective batch size = per_device * grad_accum).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate for the AdamW optimizer.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=3.0,
        help="Number of training epochs. Ignored if --max-steps > 0.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Total number of training steps. Use -1 to train by epochs.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (tokens) for each packed sample.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay applied to non-bias parameters.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for the learning rate scheduler.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Frequency of logging loss/metrics.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Frequency of checkpoint saves.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Optional evaluation frequency in steps (None disables periodic eval).",
    )
    parser.add_argument(
        "--evaluation-strategy",
        choices=("no", "steps", "epoch"),
        default="epoch",
        help="Evaluation strategy for Trainer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.05,
        help="Fraction of the dataset reserved for validation (0 disables split).",
    )
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank dimension.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha scaling.")
    parser.add_argument(
        "--lora-dropout", type=float, default=0.05, help="LoRA dropout probability."
    )
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=None,
        help=(
            "Target modules for LoRA injection. Defaults to llama-style projection layers "
            "(q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj)."
        ),
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 compute for forward/backward passes if supported.",
    )
    parser.add_argument(
        "--packing",
        action="store_true",
        help="Enable packing multiple examples within the max sequence length.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage.",
    )
    parser.add_argument(
        "--system-prompt-override",
        default=None,
        help="Override system prompt for every example (useful if base dataset lacks it).",
    )
    parser.add_argument(
        "--prefer-record-system-prompt",
        action="store_true",
        help="Use per-record system_prompt when available (ignored if override is set).",
    )

    args = parser.parse_args()
    return ScriptArgs(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        max_seq_length=args.max_seq_length,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy=args.evaluation_strategy,
        seed=args.seed,
        validation_split=args.validation_split,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bf16=args.bf16,
        packing=args.packing,
        gradient_checkpointing=args.gradient_checkpointing,
        system_prompt_override=args.system_prompt_override,
        prefer_record_system_prompt=args.prefer_record_system_prompt,
    )


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_chat_dataset(path: str) -> Dataset:
    return load_dataset("json", data_files=path, split="train")


def build_prompt_formatter(
    tokenizer: AutoTokenizer, args: ScriptArgs
):
    def formatter(example) -> List[str]:
        messages = example["messages"]

        # Apply override or per-record system prompt if requested.
        if args.system_prompt_override is not None:
            messages = [
                m for m in messages if m.get("role") != "system"
            ]
            messages.insert(
                0, {"role": "system", "content": args.system_prompt_override}
            )
        elif args.prefer_record_system_prompt and "system_prompt" in example:
            # If original dataset carried system_prompt field, ensure it's first.
            system_prompt = example.get("system_prompt")
            if system_prompt:
                messages = [
                    m for m in messages if m.get("role") != "system"
                ]
                messages.insert(0, {"role": "system", "content": system_prompt})

        text: Optional[str] = None
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            # Fallback template: simple role-tagged transcript.
            parts = []
            for message in messages:
                role = message.get("role", "user").upper()
                content = message.get("content", "")
                parts.append(f"<|{role}|>\n{content}\n</|{role}|>")
            text = "\n".join(parts)

        if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
            text += tokenizer.eos_token

        return [text]

    return formatter


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    device_map = "auto"
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        device_map=device_map,
        torch_dtype=compute_dtype,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    target_modules = args.target_modules
    if target_modules is None:
        # Default to llama-style projection layers which work for most LLaMA/OSS derivatives.
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    dataset = load_chat_dataset(args.dataset_path)

    eval_dataset = None
    if args.validation_split and 0.0 < args.validation_split < 1.0:
        split = dataset.train_test_split(test_size=args.validation_split, seed=args.seed)
        dataset = split["train"]
        eval_dataset = split["test"]

    formatter = build_prompt_formatter(tokenizer, args)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        fp16=not args.bf16,
        evaluation_strategy=args.evaluation_strategy,
        save_total_limit=3,
        load_best_model_at_end=eval_dataset is not None,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=peft_config,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatter,
        max_seq_length=args.max_seq_length,
        packing=args.packing,
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

