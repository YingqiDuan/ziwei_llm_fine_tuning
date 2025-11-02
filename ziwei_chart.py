# -*- coding: utf-8 -*-
"""
紫微斗数排盘工具（基于 py-iztro）
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Literal, Sequence

from py_iztro import Astro, AstrolabeModel
from py_iztro.core.models import PalaceModel, StarModel

# 单例 Astro 客户端，避免重复初始化 JS 虚拟机
ASTRO_CLIENT = Astro()


@dataclass
class ChartRequest:
    """排盘所需的最小输入。"""

    gender: Literal["男", "女"]
    dt: datetime  # 本地时间
    tz: float = 8.0  # 若 dt 为 naive，则按照 tz (小时) 解释
    language: Literal["en-US", "ja-JP", "ko-KR", "zh-CN", "zh-TW", "vi-VN"] = "zh-CN"
    fix_leap: bool = True  # 是否对闰月做『前半上月、后半下月』处理


def _ensure_timezone(dt: datetime, tz_offset: float) -> datetime:
    """若传入时间为 naive，则补充时区信息。"""
    if dt.tzinfo is not None:
        # 统一转回本地时区（可能是 aware 的其他时区）
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
    # 其余区间每 120 分钟一个时辰
    return ((minute - 60) // 120) + 1


def build_chart(req: ChartRequest) -> AstrolabeModel:
    """
    依据输入信息生成命盘（AstrolabeModel）。

    如果希望直接使用农历，可以改用 `Astro.by_lunar`，此函数目前仅
    提供阳历入口。若要支持其他入口，可扩展 ChartRequest 并在此处判断。
    """
    dt_local = _ensure_timezone(req.dt, req.tz)
    dt_local = dt_local.astimezone(dt_local.tzinfo)
    time_index = _time_index_from_datetime(dt_local)
    solar_date_str = f"{dt_local.year}-{dt_local.month}-{dt_local.day}"
    return ASTRO_CLIENT.by_solar(
        solar_date_str,
        time_index=time_index,
        gender=req.gender,
        fix_leap=req.fix_leap,
        language=req.language,
    )


def _format_stars(stars: Sequence[StarModel]) -> str:
    if not stars:
        return "无"
    formatted: list[str] = []
    for star in stars:
        text = star.name
        if star.brightness:
            text += f"[{star.brightness}]"
        if star.mutagen:
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
    """
    将 py-iztro 的 AstrolabeModel 渲染为多行字符串，结构与旧版保持相似。
    """
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

        assist_stars: list[StarModel] = list(palace.minor_stars)
        lines.append(f"│ │ ├辅星 : {_format_stars(assist_stars)}")

        # list(palace.adjective_stars) # 小星

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


if __name__ == "__main__":
    example = ChartRequest(
        gender="男",
        dt=datetime(2002, 3, 8, 17, 50),
        tz=8,
    )
    chart_model = build_chart(example)
    print(render_chart(chart_model))
