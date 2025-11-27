"""
Minimal QLoRA fine-tuning script for prompt/completion JSONL datasets,
but using Unsloth's FastLanguageModel for gpt-oss-20b.

Usage example:

python train_qlora_unsloth.py \
  --model-name unsloth/gpt-oss-20b-unsloth-bnb-4bit \
  --dataset-path dataset/ziwei_prompt_completion_dataset.jsonl \
  --output-dir ./outputs
"""

import argparse
import os


import unsloth
from unsloth import FastLanguageModel

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer
from trl import SFTConfig


def parse_args():
    parser = argparse.ArgumentParser("Train a QLoRA adapter with Unsloth + gpt-oss.")
    parser.add_argument("--model-name", required=True, help="Unsloth model name or local path.")
    parser.add_argument("--dataset-path", required=True, help="JSONL dataset with prompt completion pairs.")
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
        "--save-mode",
        choices=("adapter", "merged"),
        default="adapter",
        help="adapter: 只保存 LoRA 适配器 (PEFT)；merged: 保存合并后的完整 4bit 模型。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ====== 1. 用 Unsloth 加载 4bit gpt-oss ======
    max_seq_length = args.max_seq_length
    dtype = None  # 让 Unsloth 自动选 dtype（内部会处理成合适的 float32/float16 组合）

    # 这里要求 model_name 是 unsloth 提供的模型，比如：
    #  - unsloth/gpt-oss-20b-unsloth-bnb-4bit
    #  - unsloth/gpt-oss-20b  （也可以，load_in_4bit=True 会走它的 MXFP4 → NF4 逻辑）
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,      # 4bit QLoRA（Unsloth 已经封装好 MXFP4 的细节）
        full_finetuning=False,  # 我们只做 LoRA
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ====== 2. 用 Unsloth 的 LoRA 封装 ======
    # 对应你原来的 LoraConfig + get_peft_model，只是交给 Unsloth 做优化。
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,   # Unsloth 推荐 0，内核有优化
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        use_gradient_checkpointing="unsloth",  # 开启它的长上下文/省显存模式
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    model.config.use_cache = False  # 和你原来的保持一致

    # ====== 3. 数据集处理：完全沿用你原来的 prompt/completion → chunk 逻辑 ======
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

    # ====== 4. 训练参数：用 SFTConfig + 普通 Trainer ======
    # Unsloth 官方示例用的是 SFTTrainer，但因为我们已经手动处理成 input_ids / labels，
    # 用普通 Trainer + SFTConfig 也没问题。
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
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    # ====== 5. 根据 save-mode 决定怎么保存 ======

    if args.save_mode == "adapter":
        # 只保存 LoRA adapter（当前 trainer.model 就是带 LoRA 的 PEFT 模型）
        trainer.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"[train_qlora_unsloth] Saved LoRA adapter + tokenizer to {args.output_dir}")

    elif args.save_mode == "merged":
        # 保存“合并后的完整 4bit 模型”
        #
        # Unsloth 的做法是：用 merge_and_unload 把 LoRA 权重合并回基座，再保存。
        # 这样导出的目录就可以当成一个完整模型用 `from_pretrained` 或 `FastLanguageModel` 加载。
        #
        # 注意：合并过程会多占一点显存/内存，但 20B + 4bit 在 16GB 4060Ti 上一般还能撑。
        from peft import PeftModel

        # trainer.model 是 PeftModel（LoRA 包了一层）
        peft_model: PeftModel = trainer.model

        print("[train_qlora_unsloth] Merging LoRA weights into base model (this may take a while)...")
        merged_model = peft_model.merge_and_unload()  # 返回合并后的基座模型（权重已包含 LoRA）

        # 把合并后的模型保存成“完整模型目录”（仍然是 4bit，因为底层权重就是 BnB 4bit）
        merged_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        print(f"[train_qlora_unsloth] Saved merged full model + tokenizer to {args.output_dir}")

    else:
        raise ValueError(f"Unknown save_mode: {args.save_mode}")


if __name__ == "__main__":
    main()
