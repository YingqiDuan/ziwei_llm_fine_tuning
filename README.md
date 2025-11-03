# Fine-Tuning Astrology LLM – Minimal Workflow

This repo produces a Ziwei astrology chat dataset and fine‑tune a base model with QLoRA. Follow the steps below in order; each stage consumes the artifacts produced by the previous one.

## 1. Prerequisites
- Python 3.10+ 
- Install dependencies:
  ```bash
  pip install datasets peft transformers accelerate bitsandbytes trl python-dotenv certifi py-iztro
  ```
- xAI Grok API key available as `XAI_API_KEY` in your shell or a `.env` file (used by `augment.py`).

## 2. Generate Seed Ziwei Charts
Creates unique chart descriptions with random birth data.
```bash
python generate_chart.py \
  --count 500 \
  --output dataset/ziwei_chart_dataset.jsonl \
```
- Optional knobs:
  - `--start-year / --end-year` to constrain birth years.
  - `--tz` to force a fixed timezone; omit to sample from preset offsets.
- The script deduplicates charts by rendered text, so reruns with the same `--seed` plus `--count` are deterministic.

## 3. Augment With Grok Completions
Calls the xAI chat API to turn each chart into an analytical answer.
```bash
python augment.py \
  --input dataset/ziwei_chart_dataset.jsonl \
  --output dataset/ziwei_answer_dataset.jsonl \
  --model grok-4-fast \
  --limit 500 \
  --sleep 0.5
```
- `--resume` appends to an existing output file while skipping prompts already processed.
- Adjust `--system-prompt`, `--model`, and `--sleep` as needed.

## 4. Convert to Ollama Chat Format
Produces the chat-style JSONL expected by TRL’s SFT pipeline.
```bash
python extract.py \
  --input dataset/ziwei_answer_dataset.jsonl \
  --output dataset/ziwei_ollama_train.jsonl \
```
- If `--output` is omitted, the script writes to `<input>_ollama.jsonl`.
- `--system-prompt` defaults to统一的紫微顾问提示，可按需覆盖。

## 5. Fine-Tune With QLoRA
Run minimal supervised fine-tuning against the generated chat data.
```bash
python train_qlora.py \
  --model-name gpt-oss-20b \
  --dataset-path dataset/ziwei_ollama_train.jsonl \
  --output-dir outputs/gpt-oss-20b-qlora \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 2e-4 \
  --epochs 3
```
- Adjust `--model-name` to point to any compatible HF checkpoint (local or remote).
- The script loads the base model in 4-bit precision and saves only the LoRA adapter plus tokenizer.

## 6. Run Inference With the Adapter
Use the inference script to apply the fine-tuned adapter alongside the base model.
```bash
python infer.py \
  --base-model gpt-oss-20b \
  --adapter outputs/gpt-oss-20b-qlora \
  --prompt-file prompt.txt \
  --load-4bit 
```