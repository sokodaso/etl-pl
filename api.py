from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from etl.config import load_settings


class RepresentativeSong(BaseModel):
    song_key: str
    title: str
    artist: str
    chart_date: Optional[date] = None
    distance_to_centroid: float


class ClusterSummary(BaseModel):
    group_id: str
    dbscan_label: int
    cluster_size: int
    feature_summary: dict[str, Any]
    representative_songs: list[RepresentativeSong]


class ClusterSong(BaseModel):
    song_key: str
    chart_date: Optional[date] = None
    title: str
    artist: str


class ClusterDetail(ClusterSummary):
    songs: list[ClusterSong]


@lru_cache
def get_engine() -> Engine:
    mysql_url = load_settings().mysql_url
    if not mysql_url:
        raise RuntimeError("Set MYSQL_URL before starting the cluster API.")
    return create_engine(mysql_url)


def _decode_json(value: str) -> Any:
    return json.loads(value)


def _summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_id": row["stable_group_id"],
        "dbscan_label": row["dbscan_label"],
        "cluster_size": row["cluster_size"],
        "feature_summary": _decode_json(row["feature_summary_json"]),
        "representative_songs": _decode_json(row["representative_songs_json"]),
    }


ACTIVE_CLUSTERS_QUERY = text("""
    SELECT cp.stable_group_id, cp.dbscan_label, cp.cluster_size,
           cp.feature_summary_json, cp.representative_songs_json
    FROM cluster_profiles AS cp
    JOIN clustering_runs AS cr ON cr.run_id = cp.run_id
    WHERE cr.status = 'active'
    ORDER BY cp.cluster_size DESC, cp.stable_group_id
""")

ACTIVE_CLUSTER_QUERY = text("""
    SELECT cp.stable_group_id, cp.dbscan_label, cp.cluster_size,
           cp.feature_summary_json, cp.representative_songs_json
    FROM cluster_profiles AS cp
    JOIN clustering_runs AS cr ON cr.run_id = cp.run_id
    WHERE cr.status = 'active' AND cp.stable_group_id = :group_id
""")


def create_app(engine: Engine | None = None) -> FastAPI:
    application = FastAPI(title="Music Cluster API")

    def database() -> Engine:
        return engine if engine is not None else get_engine()

    @application.get("/clusters", response_model=list[ClusterSummary])
    def list_clusters(db: Engine = Depends(database)) -> list[dict[str, Any]]:
        with db.connect() as connection:
            rows = connection.execute(ACTIVE_CLUSTERS_QUERY).mappings().all()
        return [_summary(dict(row)) for row in rows]

    @application.get("/clusters/{group_id}", response_model=ClusterDetail)
    def get_cluster(group_id: str, db: Engine = Depends(database)) -> dict[str, Any]:
        with db.connect() as connection:
            profile = connection.execute(
                ACTIVE_CLUSTER_QUERY, {"group_id": group_id}
            ).mappings().first()
            if profile is None:
                raise HTTPException(status_code=404, detail="Cluster not found")
            songs = connection.execute(text("""
                SELECT sca.song_key, sca.chart_date, sca.title, sca.artist
                FROM song_cluster_assignments AS sca
                JOIN clustering_runs AS cr ON cr.run_id = sca.run_id
                WHERE cr.status = 'active' AND sca.stable_group_id = :group_id
                ORDER BY sca.chart_date DESC, sca.title, sca.artist
            """), {"group_id": group_id}).mappings().all()
        result = _summary(dict(profile))
        result["songs"] = [dict(song) for song in songs]
        return result

    return application


app = create_app()
