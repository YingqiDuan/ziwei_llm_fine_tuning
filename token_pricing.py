"""Token counting and pricing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Literal, Mapping

import tiktoken

Usage = Literal["input", "output"]
DEFAULT_TOKENIZER: Final[str] = "cl100k_base"
TOKENS_PER_MILLION: Final[int] = 1_000_000


@dataclass(frozen=True)
class TokenPrice:
    """Per-million token pricing for a single model."""

    input: float
    output: float

    def for_usage(self, usage: Usage) -> float:
        if usage not in ("input", "output"):
            raise ValueError("usage 必须是 'input' 或 'output'")
        return getattr(self, usage)


_PRICING_DATA: tuple[tuple[str, float, float], ...] = (
    ("gpt-5", 1.25, 10.0),
    ("gpt-5-mini", 0.25, 2.0),
    ("gpt-5-nano", 0.05, 0.4),
    ("gpt-5-pro", 15.0, 120.0),
    ("claude-haiku-3", 0.25, 1.25),
    ("claude-sonnet-4.5", 3.0, 15.0),
    ("claude-opus-4.1", 15.0, 75.0),
    ("grok-4-fast", 0.20, 0.50),
    ("grok-4-0709", 3.0, 15.0),
)

PRICING: Mapping[str, TokenPrice] = {
    name: TokenPrice(input_price, output_price)
    for name, input_price, output_price in _PRICING_DATA
}


@lru_cache(maxsize=None)
def _encoder_for(model: str):
    """Cache encoder lookup to avoid repeated tokenizer initialisation."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = DEFAULT_TOKENIZER) -> int:
    """Estimate tokens for `text` with the specified tokenizer model."""
    return len(_encoder_for(model).encode(text))


def calc_cost(tokens: int, model_name: str, usage: Usage = "input") -> float:
    """Convert token usage into USD cost based on the pricing table."""
    try:
        price_per_million = PRICING[model_name].for_usage(usage)
    except KeyError as exc:
        raise ValueError(f"模型 {model_name!r} 未在定价表中") from exc
    return tokens / TOKENS_PER_MILLION * price_per_million


def calc_total_cost(
    input_tokens: int,
    output_tokens: int,
    model_name: str,
) -> float:
    """Aggregate request cost covering both prompt and completion tokens."""
    return calc_cost(input_tokens, model_name, "input") + calc_cost(
        output_tokens, model_name, "output"
    )


if __name__ == "__main__":
    prompt_text = """"""
    completion_text = """"""
    model = "grok-4-fast"
    tokenizer_model = DEFAULT_TOKENIZER

    input_tokens = count_tokens(prompt_text, tokenizer_model)
    output_tokens = count_tokens(completion_text, tokenizer_model)

    print(f"输入文本: {prompt_text}")
    print(f"输出文本: {completion_text}")
    print(f"{tokenizer_model} 估算输入 token 数: {input_tokens}")
    print(f"{tokenizer_model} 估算输出 token 数: {output_tokens}")
    print(f"输入成本: ${calc_cost(input_tokens, model, 'input'):.6f}")
    print(f"输出成本: ${calc_cost(output_tokens, model, 'output'):.6f}")
    print(f"总成本: ${calc_total_cost(input_tokens, output_tokens, model):.6f}")
