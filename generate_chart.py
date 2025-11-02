"""
Minimal Ziwei chart dataset generator.
Creates random birth records, renders each chart once, and skips duplicates automatically.
"""

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from ziwei_chart import ChartRequest, build_chart, render_chart

DEFAULT_START_YEAR = 1960
DEFAULT_END_YEAR = 2024
DEFAULT_TZ_CHOICES = (
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
GENDERS = ("男", "女")


def parse_args():
    parser = argparse.ArgumentParser("Generate random Ziwei charts.")
    parser.add_argument("--count", type=int, default=100, help="Number of records to create.")
    parser.add_argument(
        "--output",
        default="dataset/ziwei_chart_dataset.jsonl",
        help="Destination JSONL file.",
    )
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--tz", type=float, default=None, help="Fixed timezone offset hours.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return parser.parse_args()


def random_datetime(rng: random.Random, start_year: int, end_year: int) -> datetime:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31, 23, 59)
    total_minutes = int((end - start).total_seconds() // 60)
    return start + timedelta(minutes=rng.randrange(total_minutes + 1))


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("count 必须为正整数")
    if args.start_year > args.end_year:
        raise ValueError("start_year 不能大于 end_year")

    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_charts: set[str] = set()
    records_written = 0

    with output_path.open("w", encoding="utf-8") as handle:
        while records_written < args.count:
            gender = rng.choice(GENDERS)
            dt = random_datetime(rng, args.start_year, args.end_year)
            tz = args.tz if args.tz is not None else rng.choice(DEFAULT_TZ_CHOICES)

            chart_model = build_chart(ChartRequest(gender=gender, dt=dt, tz=tz))
            chart_text = render_chart(chart_model)

            if chart_text in seen_charts:
                continue

            seen_charts.add(chart_text)
            payload = {
                "gender": gender,
                "birth_datetime": dt.isoformat(),
                "timezone": tz,
                "solar_date": chart_model.solar_date,
                "time_label": chart_model.time,
                "time_range": chart_model.time_range,
                "chart_text": chart_text,
            }
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
            records_written += 1

    print(f"[generate_chart] Wrote {records_written} unique records to {output_path}")


if __name__ == "__main__":
    main()
