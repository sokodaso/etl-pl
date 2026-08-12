from __future__ import annotations

import pandas as pd


def load_song_week_stats(df: pd.DataFrame, mysql_url: str, table_name: str = "song_week_stats") -> int:
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise RuntimeError(
            "MySQL loading requires SQLAlchemy and a MySQL driver. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    engine = create_engine(mysql_url)
    with engine.begin() as connection:
        df.to_sql(
            table_name,
            con=connection,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=500,
        )
    return len(df)
