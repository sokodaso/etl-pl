import json
from pathlib import Path
from typing import Any

import billboard


def extract_billboard_chart(chart_name: str = "hot-100") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    chart = billboard.ChartData(chart_name)
    raw_chart = json.loads(chart.json())

    rows = []
    for entry in chart.entries:
        rows.append(
            {
                "chart_date": chart.date,
                "rank": entry.rank,
                "title": entry.title,
                "artist": entry.artist,
                "last_rank": entry.lastPos,
                "peak_rank": entry.peakPos,
                "weeks": entry.weeks,
                "is_new": entry.isNew,
            }
        )

    return raw_chart, rows


def write_raw_billboard(raw_chart: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "billboard_hot_100.json"
    path.write_text(json.dumps(raw_chart, indent=2), encoding="utf-8")
    return path
