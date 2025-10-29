"""Generate a Ziwei dataset with varied genders and birth datetimes."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from ziwei_chart import ChartRequest, build_chart, render_chart

DEFAULT_START_YEAR = 1960
DEFAULT_END_YEAR = 2024
DEFAULT_TZ = None
GENDERS = ("男", "女")
DEFAULT_TIMEZONE_CHOICES: Sequence[float] = (
    -12,
    -11,
    -10,
    -9.5,
    -9,
    -8,
    -7,
    -6,
    -5,
    -4,
    -3.5,
    -3,
    -2,
    -1,
    0,
    0.5,
    1,
    1.5,
    2,
    2.5,
    3,
    3.5,
    4,
    4.5,
    5,
    5.5,
    5.75,
    6,
    6.5,
    7,
    7.5,
    8,
    8.75,
    9,
    9.5,
    10,
    10.5,
    11,
    11.5,
    12,
    12.75,
    13,
    14,
)


def _random_datetime(
    rng: random.Random,
    *,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
) -> datetime:
    """Pick a random datetime (minute precision) between start and end years."""
    start = datetime(start_year, 1, 1, 0, 0)
    end = datetime(end_year, 12, 31, 23, 59)
    total_minutes = int((end - start).total_seconds() // 60)
    minute_offset = rng.randrange(total_minutes + 1)
    return start + timedelta(minutes=minute_offset)


def _iter_chart_records(
    count: int,
    rng: random.Random,
    *,
    start_year: int,
    end_year: int,
    tz_picker: Callable[[], float],
) -> Iterable[dict]:
    """Yield dataset records with randomised gender and birth datetime."""
    seen: set[tuple[str, str]] = set()
    while len(seen) < count:
        gender = rng.choice(GENDERS)
        dt = _random_datetime(rng, start_year=start_year, end_year=end_year)
        key = (gender, dt.isoformat())
        if key in seen:
            continue
        seen.add(key)

        tz = tz_picker()
        chart_model = build_chart(ChartRequest(gender=gender, dt=dt, tz=tz))
        yield {
            "gender": gender,
            "birth_datetime": dt.isoformat(),
            "timezone": tz,
            "solar_date": chart_model.solar_date,
            "time_label": chart_model.time,
            "time_range": chart_model.time_range,
            "chart_text": render_chart(chart_model),
        }


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of records to generate (default: 100).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/ziwei_chart_dataset.jsonl"),
        help="Target dataset file (default: dataset/ziwei_chart_dataset.jsonl).",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        help=f"Earliest birth year (default: {DEFAULT_START_YEAR}).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
        help=f"Latest birth year (default: {DEFAULT_END_YEAR}).",
    )
    parser.add_argument(
        "--tz",
        type=float,
        default=DEFAULT_TZ,
        help="Fix timezone offset hours; if omitted, offsets are randomised.",
    )
    parser.add_argument(
        "--tz-choices",
        type=float,
        nargs="+",
        help="Custom timezone offsets for random selection (default spans UTC-12~UTC+14).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible output.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    if args.count <= 0:
        raise ValueError("count 必须为正整数")
    if args.start_year > args.end_year:
        raise ValueError("start_year 不能大于 end_year")
    if args.tz_choices and args.tz is not None:
        raise ValueError("不能同时指定固定时区 (--tz) 与随机候选 (--tz-choices)")
    if args.tz_choices:
        tz_choices: Sequence[float] = args.tz_choices
    else:
        tz_choices = DEFAULT_TIMEZONE_CHOICES
    if args.tz is None and not tz_choices:
        raise ValueError("随机时区列表为空，请使用 --tz 或提供 --tz-choices")

    rng = random.Random(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.tz is None:
        tz_picker: Callable[[], float] = lambda: rng.choice(tz_choices)
    else:
        tz_picker = lambda: args.tz

    records = _iter_chart_records(
        args.count,
        rng,
        start_year=args.start_year,
        end_year=args.end_year,
        tz_picker=tz_picker,
    )

    with args.output.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")


if __name__ == "__main__":
    main()
