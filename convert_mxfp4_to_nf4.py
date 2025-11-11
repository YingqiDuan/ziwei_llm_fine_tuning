"""
One-shot MXFP4 -> NF4 conversion utility.

Usage example:
    python convert_mxfp4_to_nf4.py \
        --model-name openai/gpt-oss-20b \
        --output-dir /path/to/local_gpt-oss-20b-nf4
"""

import argparse
import os
import shutil
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Mxfp4Config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Convert an MXFP4 checkpoint to NF4.")
    parser.add_argument(
        "--model-name",
        required=True,
        help="Source MXFP4 model name or local path.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for the NF4 checkpoint.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional attention backend (flash_attention_2, sdpa, eager, ...).",
    )
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help="Optional directory for the temporary BF16 checkpoint (default: <output>/_tmp_bf16).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow reusing an existing --output-dir by wiping it first.",
    )
    return parser.parse_args()


def ensure_clean_dir(path: str, overwrite: bool) -> None:
    if os.path.isdir(path):
        if not overwrite and os.listdir(path):
            raise RuntimeError(f"{path} already exists and is not empty; pass --overwrite to replace it.")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def attempt_model_load(model_name: str, load_kwargs: dict) -> "AutoModelForCausalLM":
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except TypeError as exc:
        if load_kwargs.get("attn_implementation") and "unexpected keyword argument" in str(exc):
            print("[convert_mxfp4_to_nf4] attn_implementation unsupported; retrying without it.")
            fallback = dict(load_kwargs)
            fallback.pop("attn_implementation", None)
            return AutoModelForCausalLM.from_pretrained(model_name, **fallback)
        raise
    except ValueError as exc:
        attn_impl = load_kwargs.get("attn_implementation")
        if attn_impl and "flash" in attn_impl.lower():
            print("[convert_mxfp4_to_nf4] Flash attention unavailable; retrying with default attention backend.")
            fallback = dict(load_kwargs)
            fallback.pop("attn_implementation", None)
            return AutoModelForCausalLM.from_pretrained(model_name, **fallback)
        raise


def save_tokenizer(model_name: str, output_dir: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.save_pretrained(output_dir)


def convert_model(
    model_name: str,
    output_dir: str,
    tmp_dir: str,
    attn_implementation: Optional[str],
) -> None:
    dequant_kwargs = {
        "device_map": "cpu",
        "quantization_config": Mxfp4Config(dequantize=True),
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if attn_implementation:
        dequant_kwargs["attn_implementation"] = attn_implementation

    print("[convert_mxfp4_to_nf4] Stage 1: dequantizing MXFP4 weights to BF16 on CPU...")
    bf16_model = attempt_model_load(model_name, dequant_kwargs)
    ensure_clean_dir(tmp_dir, overwrite=True)
    bf16_model.save_pretrained(tmp_dir, safe_serialization=True)
    del bf16_model
    torch.cuda.empty_cache()

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    reload_kwargs = {
        "device_map": "auto",
        "quantization_config": quant_config,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        reload_kwargs["attn_implementation"] = attn_implementation

    print("[convert_mxfp4_to_nf4] Stage 2: loading BF16 weights with BitsAndBytes NF4 quantization...")
    nf4_model = attempt_model_load(tmp_dir, reload_kwargs)
    print(f"[convert_mxfp4_to_nf4] Saving NF4 checkpoint to {output_dir}...")
    nf4_model.save_pretrained(output_dir, safe_serialization=True)
    del nf4_model
    torch.cuda.empty_cache()

    print("[convert_mxfp4_to_nf4] Saving tokenizer...")
    save_tokenizer(model_name, output_dir)

    try:
        shutil.rmtree(tmp_dir)
    except OSError as exc:
        print(f"[convert_mxfp4_to_nf4] Warning: failed to delete temporary directory {tmp_dir}: {exc}")


def main() -> None:
    args = parse_args()

    ensure_clean_dir(args.output_dir, overwrite=args.overwrite)
    tmp_dir = args.tmp_dir or os.path.join(args.output_dir, "_tmp_bf16")

    convert_model(
        args.model_name,
        args.output_dir,
        tmp_dir,
        args.attn_implementation,
    )
    print("[convert_mxfp4_to_nf4] Done.")


if __name__ == "__main__":
    main()
