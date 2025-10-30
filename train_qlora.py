"""
Supervised fine-tuning (SFT) entry point for gpt-oss-20B (or compatible chat models)
using QLoRA on a local Ollama-style dataset.

Example:
    python train_qlora.py \
        --model-name unsloth/gpt-oss-20b \
        --dataset-path dataset/ziwei_ollama_train.jsonl \
        --output-dir outputs/gpt-oss-20b-qlora \
        --quantization 4bit

If your checkpoint is already quantized (e.g., MXFP4), run with:
    python train_qlora.py \
        --model-name your/local/model \
        --dataset-path dataset/ziwei_ollama_train.jsonl \
        --output-dir outputs/gpt-oss-20b-qlora \
        --quantization none --full-precision-dtype auto
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

GENERATION_AWARE_TEMPLATE = """{%- for message in messages %}
{%- if message['role'] == 'system' %}
{{ '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}
{%- elif message['role'] == 'user' %}
{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}
{%- elif message['role'] == 'assistant' %}
{{ '<|im_start|>assistant\\n' }}
{% generation %}
{{ message['content'] }}
{% endgeneration %}
{{ '<|im_end|>\\n' }}
{%- elif message['role'] == 'tool' %}
{{ '<|im_start|>tool\\n' + message['content'] + '<|im_end|>\\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{ '<|im_start|>assistant\\n' }}
{% generation %}
{% endgeneration %}
{%- endif %}"""


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
    quantization: str
    full_precision_dtype: str


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
        default=4096,
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
    parser.add_argument(
        "--quantization",
        choices=("4bit", "8bit", "none"),
        default="4bit",
        help=(
            "Quantization mode for loading the base model. "
            "'4bit' enables QLoRA-style loading via bitsandbytes; "
            "'8bit' uses 8-bit weights; 'none' keeps the model's native precision "
            "(use this if your checkpoint is already quantized, e.g., MXFP4)."
        ),
    )
    parser.add_argument(
        "--full-precision-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="float16",
        help=(
            "When --quantization=none, specify the torch dtype used to load weights. "
            "Set to 'auto' to let Transformers choose."
        ),
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
        quantization=args.quantization,
        full_precision_dtype=args.full_precision_dtype,
    )


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_chat_dataset(path: str) -> Dataset:
    return load_dataset("json", data_files=path, split="train")


def apply_system_prompt_preferences(dataset: Dataset, args: ScriptArgs) -> Dataset:
    if args.system_prompt_override:
        def _override(example):
            messages = example.get("messages")
            if isinstance(messages, list):
                messages = [
                    m for m in messages if m.get("role") != "system"
                ]
                messages.insert(0, {"role": "system", "content": args.system_prompt_override})
                example["messages"] = messages
            return example

        return dataset.map(_override)

    if args.prefer_record_system_prompt:
        def _prefer(example):
            system_prompt = example.get("system_prompt")
            messages = example.get("messages")
            if system_prompt and isinstance(messages, list):
                messages = [
                    m for m in messages if m.get("role") != "system"
                ]
                messages.insert(0, {"role": "system", "content": system_prompt})
                example["messages"] = messages
            return example

        return dataset.map(_prefer)

    return dataset


def chat_template_supports_assistant_masks(tokenizer: AutoTokenizer) -> bool:
    if not hasattr(tokenizer, "apply_chat_template"):
        return False

    sample = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    try:
        processed = tokenizer.apply_chat_template(
            sample,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
    except Exception:
        return False

    assistant_masks = processed.get("assistant_masks")
    if assistant_masks is None:
        return False

    if isinstance(assistant_masks, list):
        if assistant_masks and isinstance(assistant_masks[0], list):
            assistant_masks = assistant_masks[0]
        return any(bool(x) for x in assistant_masks)

    try:
        # tensors or numpy arrays
        return bool(assistant_masks.any())
    except Exception:
        return False


def ensure_generation_keyword(tokenizer: AutoTokenizer) -> None:
    if chat_template_supports_assistant_masks(tokenizer):
        return

    print(
        "[train_qlora] Updating tokenizer chat_template to include `{% generation %}` for assistant-only loss."
    )
    tokenizer.chat_template = GENERATION_AWARE_TEMPLATE

    if not chat_template_supports_assistant_masks(tokenizer):
        raise RuntimeError(
            "Failed to configure a generation-aware chat template. "
            "Please provide a compatible template or adjust training settings."
        )


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    device_map = "auto"
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16

    config = None
    try:
        config = AutoConfig.from_pretrained(
            args.model_name,
            trust_remote_code=True,
        )
    except Exception as exc:
        print(
            f"[train_qlora] Warning: unable to inspect base config for quantization"
            f" ({exc}). Proceeding with CLI-provided flags."
        )

    existing_quant_config = getattr(config, "quantization_config", None) if config else None
    effective_quantization = args.quantization
    should_set_dtype = args.full_precision_dtype != "auto"

    if existing_quant_config is not None:
        quant_name = getattr(existing_quant_config, "quant_method", None)
        if quant_name is None:
            quant_name = getattr(existing_quant_config, "quantization_method", None)
        if quant_name is None and hasattr(existing_quant_config, "to_dict"):
            quant_name = existing_quant_config.to_dict().get("quant_method")
        if quant_name is None and isinstance(existing_quant_config, dict):
            quant_name = existing_quant_config.get("quant_method")
        if quant_name is None:
            quant_name = type(existing_quant_config).__name__

        if args.quantization != "none":
            print(
                "[train_qlora] Detected existing quantization"
                f" ({quant_name}); ignoring --quantization={args.quantization} to avoid"
                " configuration conflicts. Re-run with --quantization none to silence this"
                " warning."
            )
        effective_quantization = "none"
        should_set_dtype = False

    quant_config = None
    if effective_quantization == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif effective_quantization == "8bit":
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_has_fp16_weight=False,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ensure_generation_keyword(tokenizer)

    model_kwargs = {
        "device_map": device_map,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["dtype"] = compute_dtype
    elif should_set_dtype:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        model_kwargs["dtype"] = dtype_map[args.full_precision_dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if effective_quantization in {"4bit", "8bit"}:
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
    dataset = apply_system_prompt_preferences(dataset, args)

    eval_dataset = None
    if args.validation_split and 0.0 < args.validation_split < 1.0:
        split = dataset.train_test_split(test_size=args.validation_split, seed=args.seed)
        dataset = split["train"]
        eval_dataset = split["test"]
        print(
            f"[train_qlora] Dataset split into {len(dataset)} train and {len(eval_dataset)} eval samples "
            f"(validation_split={args.validation_split})."
        )
    else:
        print(f"[train_qlora] Loaded dataset with {len(dataset)} samples (no validation split).")

    training_args = SFTConfig(
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
        eval_strategy=args.evaluation_strategy,
        save_total_limit=3,
        load_best_model_at_end=eval_dataset is not None,
        report_to="none",
        seed=args.seed,
        packing=args.packing,
        max_length=args.max_seq_length,
        assistant_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
    )

    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"[train_qlora] Trainable parameter count: {trainable:,}")

    print(
        "[train_qlora] Starting training "
        f"(epochs={training_args.num_train_epochs}, max_steps={training_args.max_steps}, "
        f"batch_size={training_args.per_device_train_batch_size}, grad_accum={training_args.gradient_accumulation_steps})."
    )

    trainer.train()
    print("[train_qlora] Training complete. Saving adapter and tokenizer...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[train_qlora] Saved artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
