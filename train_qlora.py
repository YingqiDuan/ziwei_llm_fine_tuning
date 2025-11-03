"""
Minimal QLoRA fine-tuning script for prompt/completion JSONL datasets.
Only the essentials are retained so you can point to a model, a dataset, and an output path.
"""

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer
from trl import SFTConfig


def parse_args():
    parser = argparse.ArgumentParser("Train a QLoRA adapter with minimal knobs.")
    parser.add_argument("--model-name", required=True, help="Base model name or local path.")
    parser.add_argument("--dataset-path", required=True, help="JSONL dataset with prompt/completion pairs.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the adapter/tokenizer.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            use_fast=True,
            trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        quantization_config=quant_config,
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    tokenizer_max_length = getattr(tokenizer, "model_max_length", None)
    max_seq_length = int(tokenizer_max_length)

    def split_prompt_completion(example):
        prompt_text = example.get("prompt", "").strip()
        completion_text = example.get("completion", "").strip()
        if not completion_text:
            raise ValueError("Each example must include a completion.")
        if not prompt_text:
            raise ValueError("Each example must include a prompt.")

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids

        if len(input_ids) > max_seq_length:
            input_ids = input_ids[-max_seq_length:]
            labels = labels[-max_seq_length:]

        attention_mask = [1] * len(input_ids)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    dataset = dataset.map(split_prompt_completion, remove_columns=dataset.column_names)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    sample_batch = next(iter(trainer.get_train_dataloader()))
    labels = sample_batch["labels"]
    valid_mask = labels != -100
    per_sample_valid = valid_mask.sum(dim=1)
    print(
        "[train_qlora] Assistant-token counts per sample in first batch:",
        per_sample_valid.tolist(),
    )
    print(
        "[train_qlora] Mean assistant tokens this batch:",
        per_sample_valid.float().mean().item(),
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[train_qlora] Saved adapter and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
