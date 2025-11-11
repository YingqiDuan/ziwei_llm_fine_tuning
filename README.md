# Fine-Tuning Astrology LLM – Minimal Workflow

This repo produces a Ziwei astrology chat dataset and fine‑tune a base model with QLoRA. Follow the steps below in order; each stage consumes the artifacts produced by the previous one.

## 1. Prerequisites
- Python 3.10+ 
- Install dependencies:
```powershell
pip install datasets peft transformers accelerate bitsandbytes trl python-dotenv certifi py-iztro
```
- xAI Grok API key available as `XAI_API_KEY` in your shell or a `.env` file (used by `augment.py`).

## 2. Generate Seed Ziwei Charts
Creates unique chart descriptions with random birth data.
```powershell
python .\generate_chart.py `
  --count 500 `
  --output .\dataset\ziwei_chart_dataset.jsonl
```
- Optional knobs:
  - `--start-year / --end-year` to constrain birth years.
  - `--tz` to force a fixed timezone; omit to sample from preset offsets.
- The script deduplicates charts by rendered text, so reruns with the same `--seed` plus `--count` are deterministic.

## 3. Augment With Grok Completions
Calls the xAI chat API to turn each chart into an analytical answer.
```powershell
python .\augment.py `
  --input .\dataset\ziwei_chart_dataset.jsonl `
  --output .\dataset\ziwei_answer_dataset.jsonl `
  --model grok-4-fast `
  --limit 500 `
  --sleep 0.5
```
- `--resume` appends to an existing output file while skipping prompts already processed.
- Adjust `--system-prompt`, `--model`, and `--sleep` as needed.

## 4. Flatten Prompt/Completion Pairs
Produces a simplified JSONL with `prompt` and `completion` fields for supervised fine-tuning.
```powershell
python .\extract.py `
  --input .\dataset\ziwei_answer_dataset.jsonl `
  --output .\dataset\ziwei_answer_dataset_prompt_completion.jsonl
```

## 5. Convert MXFP4 Checkpoints to NF4 (one-time)
Run the conversion script once per MXFP4 base model so future QLoRA jobs can load the already quantized NF4 checkpoint directly.
```powershell
python .\convert_mxfp4_to_nf4.py `
  --model-name openai/gpt-oss-20b `
  --output-dir .\models\gpt-oss-20b-nf4 `
  --attn-implementation flash_attention_2
```
- The script upcasts the MXFP4 weights to BF16 on CPU, quantizes them back to 4-bit NF4 with BitsAndBytes, then calls `save_pretrained` on the result plus tokenizer.
- Expect ~60 GB of host RAM (or swap) during the temporary BF16 stage; delete `models/gpt-oss-20b-nf4/_tmp_bf16` if an interrupted run leaves it behind.
- WSL tip: set a generous memory ceiling and swap file in `%UserProfile%\.wslconfig`, then run `wsl --shutdown` so the new limits apply, e.g.
  ```
  [wsl2]
  memory=60GB
  processors=12
  swap=128GB
  swapFile=C:\\wsl-swap.vhdx
  ```
  WSL only uses swap after hitting the `memory` ceiling, so choose a value close to but below your physical RAM (e.g. 60 GB on a 64 GB host).
- Use `--overwrite` when re-running the conversion into the same directory.
- Point the `train_qlora.py --model-name` argument to the NF4 directory you just produced (e.g. `models/gpt-oss-20b-nf4`).
- Shortcut: if you cannot covert, skip this step entirely and load `mdouglas/gpt-oss-20b-bnb-nf4` (or any other NF4-ready repo) directly via `--model-name`.

## 6. Fine-Tune With QLoRA
Run minimal supervised fine-tuning against the flattened prompt/completion data.
```powershell
python .\train_qlora.py `
  --model-name .\models\gpt-oss-20b-nf4 `
  --dataset-path .\dataset\ziwei_answer_dataset_prompt_completion.jsonl `
  --output-dir .\outputs\gpt-oss-20b-qlora `
  --max-seq-length 2048 `
  --context-keep 256 `
  --attn-implementation flash_attention_2 `
  --batch-size 1 `
  --grad-accum 8 `
  --learning-rate 2e-4 `
  --epochs 3
```
- Adjust `--model-name` to point to any compatible HF checkpoint, but using the locally converted NF4 directory keeps future runs fast and memory efficient.
- MXFP4 checkpoints such as `openai/gpt-oss-20b` are still detected automatically for backward compatibility. If you skip the explicit conversion step, the training script falls back to on-the-fly MXFP4 → BF16 → NF4 conversion (which costs extra time and RAM per run).
- `--max-seq-length` controls chunk size; each chunk repeats the full prompt and fits within this length (default 2048 tokens).
- `--context-keep` retains that many completion tokens from the previous chunk as loss-masked context (default 256).
- `--attn-implementation` defaults to `flash_attention_2`; switch to `sdpa`/`eager` if FlashAttention is unavailable on your setup.
- The trainer uses the `paged_adamw_8bit` optimizer, loads the base model in 4-bit precision, and saves only the LoRA adapter plus tokenizer.

## 7. Run Inference With the Adapter
Use the inference script to apply the fine-tuned adapter alongside the base model.
```powershell
python .\infer.py `
  --base-model gpt-oss-20b `
  --adapter .\outputs\gpt-oss-20b-qlora `
  --prompt-file .\prompt.txt `
  --load-4bit
```
