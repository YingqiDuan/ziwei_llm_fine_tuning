"""
Minimal QLoRA fine-tuning script for prompt/completion JSONL datasets.
Only the essentials are retained so you can point to a model, a dataset, and an output path.
"""

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    Mxfp4Config,
)
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
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum tokens per training sample after chunking.",
    )
    parser.add_argument(
        "--context-keep",
        type=int,
        default=256,
        help="Number of previous tokens to retain as context between chunks (masked from loss).",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="Attention backend passed to from_pretrained (e.g. flash_attention_2, sdpa, eager).",
    )
    return parser.parse_args()


def attempt_model_load(model_name: str, load_kwargs: dict) -> "AutoModelForCausalLM":
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except TypeError as exc:
        if load_kwargs.get("attn_implementation") and "unexpected keyword argument" in str(exc):
            print(
                "[train_qlora] attn_implementation unsupported by this model class; falling back to default.",
            )
            fallback = dict(load_kwargs)
            fallback.pop("attn_implementation", None)
            return AutoModelForCausalLM.from_pretrained(model_name, **fallback)
        raise
    except ValueError as exc:
        attn_impl = load_kwargs.get("attn_implementation")
        if attn_impl and "flash" in attn_impl.lower():
            print(
                "[train_qlora] Flash attention unavailable for this model; reverting to default attention backend.",
            )
            fallback = dict(load_kwargs)
            fallback.pop("attn_implementation", None)
            return AutoModelForCausalLM.from_pretrained(model_name, **fallback)
        raise


def is_mxfp4_quantized(model_name: str) -> bool:
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception as exc:
        print(f"[train_qlora] Warning: failed to inspect config for {model_name}: {exc}")
        return False

    quant_cfg = getattr(config, "quantization_config", None)

    if isinstance(quant_cfg, dict):
        quant_method = quant_cfg.get("quant_method") or quant_cfg.get("quantization_method")
        return isinstance(quant_method, str) and quant_method.lower() == "mxfp4"

    return False


def load_mxfp4_model_as_nf4(
    model_name: str,
    quant_config: BitsAndBytesConfig,
    attn_implementation: str | None,
) -> "AutoModelForCausalLM":
    # Stage 1: dequantize MXFP4 weights to BF16 on CPU.
    dequant_kwargs = {
        "device_map": "cpu",
        "quantization_config": Mxfp4Config(dequantize=True),
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if attn_implementation:
        dequant_kwargs["attn_implementation"] = attn_implementation

    base_model = attempt_model_load(model_name, dequant_kwargs)
    state_dict = base_model.state_dict()
    del base_model
    torch.cuda.empty_cache()

    # Stage 2: reload using BitsAndBytes 4-bit quantization.
    reload_kwargs = {
        "device_map": "auto",
        "quantization_config": quant_config,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        reload_kwargs["attn_implementation"] = attn_implementation

    model = attempt_model_load(model_name, {**reload_kwargs, "state_dict": state_dict})
    state_dict.clear()
    torch.cuda.empty_cache()

    return model


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

    if is_mxfp4_quantized(args.model_name):
        print(
            "[train_qlora] Detected MXFP4 checkpoint; dequantizing to BF16 then re-quantizing to NF4 for QLoRA.",
        )
        model = load_mxfp4_model_as_nf4(args.model_name, quant_config, args.attn_implementation)
    else:
        model_kwargs: dict[str, object] = {
            "device_map": "auto",
            "quantization_config": quant_config,
            "trust_remote_code": True,
        }
        if args.attn_implementation:
            model_kwargs["attn_implementation"] = args.attn_implementation
        model = attempt_model_load(args.model_name, model_kwargs)

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
    requested_max_seq_len = args.max_seq_length
    if tokenizer_max_length is None or tokenizer_max_length <= 0:
        max_seq_length = requested_max_seq_len
    else:
        max_seq_length = min(requested_max_seq_len, int(tokenizer_max_length))

    def split_prompt_completion(batch):
        prompts = batch.get("prompt")
        completions = batch.get("completion")
        if prompts is None or completions is None:
            raise ValueError("Dataset records must include prompt and completion fields.")

        output_input_ids: list[list[int]] = []
        output_attention: list[list[int]] = []
        output_labels: list[list[int]] = []

        for prompt_text, completion_text in zip(prompts, completions):
            completion_text = (completion_text or "").strip()
            if not completion_text:
                continue
            prompt_text = (prompt_text or "").strip()

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]

            prompt_len = len(prompt_ids)
            if prompt_len >= max_seq_length:
                raise ValueError(
                    "Prompt length exceeds or equals max-seq-length; cannot include completion tokens."
                )

            chunk_capacity = max_seq_length - prompt_len
            prompt_label_prefix = [-100] * prompt_len

            completion_start = 0
            total_completion = len(completion_ids)
            while completion_start < total_completion:
                ctx_len = min(args.context_keep, completion_start)
                if ctx_len >= chunk_capacity:
                    ctx_len = max(0, chunk_capacity - 1)

                new_len_available = chunk_capacity - ctx_len
                if new_len_available <= 0:
                    # No room left for new tokens; force at least one token by reducing context.
                    ctx_len = 0
                    new_len_available = chunk_capacity
                    if new_len_available <= 0:
                        break

                new_len = min(new_len_available, total_completion - completion_start)
                if new_len <= 0:
                    break

                overlap_start = completion_start - ctx_len
                new_end = completion_start + new_len

                chunk_completion_ids = completion_ids[overlap_start:new_end]

                chunk_input_ids = prompt_ids + chunk_completion_ids
                chunk_attention = [1] * len(chunk_input_ids)

                chunk_labels = prompt_label_prefix.copy()
                chunk_labels.extend([-100] * ctx_len)
                chunk_labels.extend(completion_ids[completion_start:new_end])

                output_input_ids.append(chunk_input_ids)
                output_attention.append(chunk_attention)
                output_labels.append(chunk_labels)

                completion_start = new_end

        return {
            "input_ids": output_input_ids,
            "attention_mask": output_attention,
            "labels": output_labels,
        }

    dataset = dataset.map(
        split_prompt_completion,
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=1,
    )

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
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[train_qlora] Saved adapter and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
