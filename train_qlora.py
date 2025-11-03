"""
Minimal QLoRA fine-tuning script for chat-style JSONL datasets.
Only the essentials are retained so you can point to a model, a dataset, and an output path.
"""

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser("Train a QLoRA adapter with minimal knobs.")
    parser.add_argument("--model-name", required=True, help="Base model name or local path.")
    parser.add_argument("--dataset-path", required=True, help="JSONL dataset in Ollama chat format.")
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
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # TRL's assistant-only masking requires the chat template to include the
    # `{% generation %}` marker. Older Qwen releases omit it, so we patch in a
    # minimal template compatible with the model's conversation format.
    needs_generation_marker = "{% generation %}" not in (tokenizer.chat_template or "")
    if needs_generation_marker and "qwen" in args.model_name.lower():
        tokenizer.chat_template = (
            "{{ bos_token or '' }}\n"
            "{% for message in messages %}\n"
            "{% if message['role'] == 'system' %}"
            "{{ '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{% elif message['role'] == 'tool' %}"
            "{{ '<|im_start|>tool\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{% else %}"
            "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{% endif %}\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}\n"
            "{% generation %}\n"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% endif %}"
        )

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

    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        assistant_only_loss=True,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        fp16=True,
        bf16=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=peft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[train_qlora] Saved adapter and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
