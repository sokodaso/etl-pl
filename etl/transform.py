from __future__ import annotations

import pandas as pd


CHART_COLUMNS = [
    "chart_date",
    "rank",
    "title",
    "artist",
    "last_rank",
    "peak_rank",
    "weeks",
    "is_new",
]


YOUTUBE_COLUMNS = [
    "youtube_video_id",
    "youtube_title",
    "youtube_channel",
    "youtube_published_at",
    "youtube_views",
    "youtube_likes",
    "youtube_comments",
    "youtube_search_query",
]


GENIUS_COLUMNS = [
    "genius_song_id",
    "genius_pageviews",
    "genius_annotation_count",
    "genius_url",
]

KEY_COLUMNS = ["chart_date", "rank", "title", "artist"]


def build_song_week_stats(
    chart_rows: list[dict],
    youtube_rows: list[dict],
    genius_rows: list[dict],
) -> pd.DataFrame:
    chart_df = pd.DataFrame(chart_rows)
    youtube_df = pd.DataFrame(youtube_rows)
    genius_df = pd.DataFrame(genius_rows)

    for dataframe in (youtube_df, genius_df):
        for column in KEY_COLUMNS:
            if column not in dataframe.columns:
                dataframe[column] = None

    for column in CHART_COLUMNS:
        if column not in chart_df.columns:
            chart_df[column] = None

    df = chart_df[CHART_COLUMNS].copy()
    df = df.merge(youtube_df, how="left", on=KEY_COLUMNS)
    df = df.merge(genius_df, how="left", on=KEY_COLUMNS)

    for column in YOUTUBE_COLUMNS + GENIUS_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df["chart_date"] = pd.to_datetime(df["chart_date"]).dt.date
    df["youtube_published_at"] = pd.to_datetime(df["youtube_published_at"], errors="coerce", utc=True)
    df["is_new"] = df["is_new"].astype(bool)

    return df[
        CHART_COLUMNS
        + YOUTUBE_COLUMNS
        + GENIUS_COLUMNS
    ].sort_values(["chart_date", "rank"])
