from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Literal, Sequence

import boto3
from botocore.exceptions import ClientError

from py_iztro import Astro, AstrolabeModel
from py_iztro.core.models import PalaceModel, StarModel

# =========================
# 0) Bedrock 配置
# =========================
REGION = "us-east-1"
MODEL_ARN = "arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/xxxxxxxx"  

# =========================
# 1) 紫微斗数排盘
# =========================

ASTRO_CLIENT = Astro()

DEFAULT_SYSTEM_PROMPT = (
    "你现在是资深的国学易经术数领域专家，请详细分析下面这个紫微斗数命盘，"
    "综合使用三合紫微、飞星紫微、钦天四化等各流派紫微斗数的分析技法，对命盘十二宫星曜分布和各宫位间的飞宫四化进行细致分析，"
    "进而对命主的健康、学业、事业、财运、人际关系、婚姻和感情等各个方面进行全面分析和总结。"
)


@dataclass
class ChartRequest:
    gender: Literal["male", "female"]
    dt: datetime  
    tz: float = 0.0  


def _ensure_timezone(dt: datetime, tz_offset: float) -> datetime:
    """若传入时间为 naive，则补充 tzinfo（不做时区换算，仅最小补全）。"""
    if dt.tzinfo is not None:
        return dt
    tz = timezone(timedelta(hours=tz_offset))
    return dt.replace(tzinfo=tz)


def _time_index_from_datetime(dt: datetime) -> int:
    """将具体时间换算为 py-iztro 所需的时辰序号（0~12）。"""
    minute = dt.hour * 60 + dt.minute
    if minute < 0 or minute >= 1440:
        raise ValueError(f"非法时间分钟数: {minute}")
    if minute < 60:
        return 0  # 00:00~01:00 早子时
    if minute >= 1380:
        return 12  # 23:00~24:00 晚子时
    return ((minute - 60) // 120) + 1


def build_chart(req: ChartRequest) -> AstrolabeModel:
    dt_local = _ensure_timezone(req.dt, req.tz)
    dt_local = dt_local.astimezone(dt_local.tzinfo)  

    time_index = _time_index_from_datetime(dt_local)
    solar_date_str = f"{dt_local.year}-{dt_local.month}-{dt_local.day}"

    gender_zh = "男" if req.gender == "male" else "女"

    return ASTRO_CLIENT.by_solar(
        solar_date_str,
        time_index=time_index,
        gender=gender_zh,
        fix_leap=True,       
        language="zh-CN",     
    )


def _format_stars(stars: Sequence[StarModel]) -> str:
    if not stars:
        return "无"
    formatted: list[str] = []
    for star in stars:
        text = star.name
        if getattr(star, "brightness", None):
            text += f"[{star.brightness}]"
        if getattr(star, "mutagen", None):
            text += f"[{star.mutagen}]"
        formatted.append(text)
    return ",".join(formatted)


def _iter_palaces_in_ming_order(palaces: Sequence[PalaceModel]) -> Iterable[PalaceModel]:
    """从命宫开始顺序输出十二宫。"""
    if not palaces:
        return []
    ming_index = next((i for i, p in enumerate(palaces) if p.name == "命宫"), 0)
    count = len(palaces)
    for offset in range(count):
        yield palaces[(ming_index + offset) % count]


def render_chart(model: AstrolabeModel) -> str:
    """将 py-iztro 的 AstrolabeModel 渲染为多行字符串。"""
    lines: list[str] = []
    lines.append("紫微斗数命盘")
    lines.append("│")
    lines.append("├基本信息")
    lines.append("│ │")
    lines.append(f"│ ├性别 : {model.gender}")
    lines.append(f"│ ├阳历 : {model.solar_date} {model.time} ({model.time_range})")
    lines.append(f"│ ├农历 : {model.lunar_date}")
    lines.append(f"│ ├干支 : {model.chinese_date}")
    lines.append(f"│ ├命主 : {model.soul}")
    lines.append(f"│ ├身主 : {model.body}")
    lines.append(f"│ ├命宫地支 : {model.earthly_branch_of_soul_palace}")
    lines.append(f"│ ├身宫地支 : {model.earthly_branch_of_body_palace}")
    lines.append(f"│ └五行局 : {model.five_elements_class}")
    lines.append("│")
    lines.append("├命盘十二宫")
    lines.append("│ │")

    for palace in _iter_palaces_in_ming_order(model.palaces):
        tags: list[str] = []
        if palace.is_body_palace:
            tags.append("身宫")

        head = f"│ ├{palace.name}[{palace.heavenly_stem}{palace.earthly_branch}]"
        if tags:
            head += "[" + "、".join(tags) + "]"
        lines.append(head)

        lines.append(f"│ │ ├主星 : {_format_stars(palace.major_stars)}")
        lines.append(f"│ │ ├辅星 : {_format_stars(list(palace.minor_stars))}")

        if palace.decadal:
            start, end = palace.decadal.range
            lines.append(
                f"│ │ ├大限 : {start}~{end}虚岁 "
                f"({palace.decadal.heavenly_stem}{palace.decadal.earthly_branch})"
            )
        else:
            lines.append("│ │ ├大限 : 未提供")

        if palace.ages:
            preview = ",".join(str(age) for age in palace.ages[:5])
            suffix = "…" if len(palace.ages) > 5 else ""
            lines.append(f"│ │ └流年 : {preview}{suffix}")
        else:
            lines.append("│ │ └流年 : 无")

        lines.append("│ │")

    lines.append("│")
    lines.append("└")
    return "\n".join(lines)


# =========================
# 2) Bedrock invoke_model
# =========================

def extract_reasoning_content(result: dict) -> str:
    """
    只取 OpenAI 风格返回中的 reasoning_content：
      {"choices":[{"message":{"reasoning_content":"..."}}]}
    """
    choices = result.get("choices") or []
    if not choices:
        return json.dumps(result, ensure_ascii=False, indent=2)

    msg = (choices[0] or {}).get("message") or {}
    reasoning = msg.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    return json.dumps(result, ensure_ascii=False, indent=2)



def call_bedrock(user_content: str, temperature: float = 0.0, top_p: float = 1.0) -> tuple[str, dict]:
    bedrock = boto3.client("bedrock-runtime", region_name=REGION)

    body = {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "top_p": top_p,
    }

    max_retries = 10
    backoff = 5

    for attempt in range(1, max_retries + 1):
        try:
            response = bedrock.invoke_model(
                modelId=MODEL_ARN,
                body=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            return extract_reasoning_content(result), result

        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code == "ModelNotReadyException":
                print(f"[attempt {attempt}] Model not ready, sleep {backoff}s ...")
                time.sleep(backoff)
                backoff *= 2
                continue
            raise

    raise RuntimeError("Model still not ready after retries")


# =========================
# 3) CLI
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gender", required=True, choices=["male", "female"])
    p.add_argument("--datetime", required=True, help='format: "YYYY-MM-DD HH:MM"')
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dt = datetime.strptime(args.datetime, "%Y-%m-%d %H:%M")

    req = ChartRequest(gender=args.gender, dt=dt)
    chart_model = build_chart(req)
    chart_text = render_chart(chart_model)

    text, raw = call_bedrock(
        user_content=chart_text,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(text)
    if isinstance(raw.get("usage"), dict):
        print("usage:", raw["usage"])


if __name__ == "__main__":
    main()