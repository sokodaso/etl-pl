from __future__ import annotations

import argparse
import json
from pathlib import Path

from etl.billboard_client import extract_billboard_chart, write_raw_billboard
from etl.genius_client import build_genius_client, search_song_metadata
from etl.youtube_client import build_youtube_client, search_top_video
from etl.config import load_settings
from etl.loader import load_song_week_stats
from etl.transform import build_song_week_stats



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weekly Billboard Hot 100 song metrics.")
    parser.add_argument("--chart", default=None, help="Billboard chart slug. Defaults to BILLBOARD_CHART or hot-100.")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for development/API testing.")
    parser.add_argument("--skip-youtube", action="store_true", help="Skip YouTube enrichment.")
    parser.add_argument("--skip-genius", action="store_true", help="Skip Genius enrichment.")
    parser.add_argument("--skip-load", action="store_true", help="Write files but do not load MySQL.")
    parser.add_argument("--output-dir", default=None, help="Directory for raw JSON and final CSV outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    if not settings.mysql_url and not args.skip_load:
        raise ValueError("Set MYSQL_URL or run with --skip-load.")
    chart_name = args.chart or settings.chart_name
    output_dir = Path(args.output_dir or settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_chart, chart_rows = extract_billboard_chart(chart_name)
    if args.limit:
        chart_rows = chart_rows[: args.limit]
    write_raw_billboard(raw_chart, output_dir)

    youtube_rows = enrich_youtube(chart_rows, settings, output_dir, skip=args.skip_youtube)
    genius_rows = enrich_genius(chart_rows, settings, output_dir, skip=args.skip_genius)

    stats_df = build_song_week_stats(chart_rows, youtube_rows, genius_rows)
    csv_path = output_dir / "song_week_stats.csv"
    stats_df.to_csv(csv_path, index=False)

    if not args.skip_load:
        if not settings.mysql_url:
            raise ValueError("Set MYSQL_URL or run with --skip-load.")
        loaded_count = load_song_week_stats(stats_df, settings.mysql_url)
        print(f"Loaded {loaded_count} rows into MySQL table song_week_stats.")

    print(f"Wrote {len(stats_df)} rows to {csv_path}.")


def enrich_youtube(chart_rows: list[dict], settings, output_dir: Path, skip: bool = False) -> list[dict]:
    if skip:
        return []

    youtube = build_youtube_client(settings.youtube_api_key, settings.youtube_client_secret_path)
    enriched_rows = []

    # Check for cached YouTube enrichment to avoid repeated API calls during development
    cache_path = output_dir / "youtube_enrichment.json"
    if cache_path.exists():
        try:
            cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
            cache = {(_song_key(row)["artist"], _song_key(row)["title"]): row for row in cached_data}
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode cached YouTube enrichment from {cache_path}.")
   
    for row in chart_rows:
        lookup_key = (_song_key(row)["artist"], _song_key(row)["title"])
        if cache_path.exists() and lookup_key in cache:
            enriched_rows.append(cache[lookup_key])
        else:
            print(f"YouTube: {row['artist']} - {row['title']}")
            metadata = search_top_video(youtube, row["artist"], row["title"]) or {}
            enriched_rows.append({**_song_key(row), **metadata})
            cache[lookup_key] = enriched_rows[-1]

    _write_json(enriched_rows, output_dir / "youtube_enrichment.json")
    return enriched_rows


def enrich_genius(chart_rows: list[dict], settings, output_dir: Path, skip: bool = False) -> list[dict]:
    if skip:
        return []

    genius = build_genius_client(settings.genius_access_token)
    enriched_rows = []

    # Cache Genius enrichment to avoid repeated API calls during development
    cache_path = output_dir / "genius_enrichment.json"
    if cache_path.exists():
        try:
            cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
            cache = {(_song_key(row)["artist"], _song_key(row)["title"]): row for row in cached_data}
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode cached Genius enrichment from {cache_path}.")
    for row in chart_rows:
        lookup_key = (_song_key(row)["artist"], _song_key(row)["title"])
        if cache_path.exists() and lookup_key in cache:
            enriched_rows.append(cache[lookup_key])
        else:
            print(f"Genius: {row['artist']} - {row['title']}")
            metadata = search_song_metadata(genius, row["artist"], row["title"]) or {}
            enriched_rows.append({**_song_key(row), **metadata})
            cache[lookup_key] = enriched_rows[-1]

    _write_json(enriched_rows, output_dir / "genius_enrichment.json")
    return enriched_rows


def _song_key(row: dict) -> dict:
    return {
        "chart_date": row["chart_date"],
        "rank": row["rank"],
        "title": row["title"],
        "artist": row["artist"],
    }


def _write_json(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
