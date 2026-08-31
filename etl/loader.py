from __future__ import annotations

import pandas as pd


def prepare_for_sql(df: pd.DataFrame) -> pd.DataFrame:
    """Return SQL-safe values, representing pandas missing values as NULL."""
    return df.astype(object).where(pd.notna(df), None)


def load_song_week_stats(df: pd.DataFrame, mysql_url: str, table_name: str = "song_week_stats") -> int:
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise RuntimeError(
            "MySQL loading requires SQLAlchemy and a MySQL driver. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    sql_df = prepare_for_sql(df)
    engine = create_engine(mysql_url)
    with engine.begin() as connection:
        sql_df.to_sql(
            table_name,
            con=connection,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=500,
        )
    return len(df)
