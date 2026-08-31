from datetime import date, datetime, timezone
import json

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from api import create_app
from cluster import ensure_clustering_tables


def api_client() -> TestClient:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    ensure_clustering_tables(engine)
    with engine.begin() as connection:
        connection.execute(text("""
            INSERT INTO clustering_runs
            (run_id, created_at, status, dataset_size, cluster_count, noise_count,
             noise_ratio, parameters_json, metrics_json, artifact_path)
            VALUES ('run-one', :created_at, 'active', 1, 1, 0, 0, '{}', '{}', '/tmp/model')
        """), {"created_at": datetime.now(timezone.utc)})
        connection.execute(text("""
            INSERT INTO music_groups
            (stable_group_id, created_run_id, last_seen_run_id, status)
            VALUES ('group_abc', 'run-one', 'run-one', 'active')
        """))
        connection.execute(text("""
            INSERT INTO cluster_profiles
            (run_id, stable_group_id, dbscan_label, cluster_size, centroid_pca_json,
             feature_summary_json, representative_songs_json)
            VALUES ('run-one', 'group_abc', 0, 1, '[]', :summary, :representatives)
        """), {
            "summary": json.dumps({"rank": {"mean": 1.0}}),
            "representatives": json.dumps([{
                "song_key": "song-key", "title": "Song", "artist": "Artist",
                "chart_date": "2026-08-01", "distance_to_centroid": 0.0,
            }]),
        })
        connection.execute(text("""
            INSERT INTO song_cluster_assignments
            (run_id, song_key, chart_date, title, artist, dbscan_label,
             stable_group_id, is_noise)
            VALUES ('run-one', 'song-key', :chart_date, 'Song', 'Artist', 0, 'group_abc', 0)
        """), {"chart_date": date(2026, 8, 1)})
    return TestClient(create_app(engine))


def test_list_clusters_returns_active_profiles():
    response = api_client().get("/clusters")

    assert response.status_code == 200
    assert response.json()[0]["group_id"] == "group_abc"
    assert response.json()[0]["cluster_size"] == 1


def test_get_cluster_returns_profile_and_songs():
    response = api_client().get("/clusters/group_abc")

    assert response.status_code == 200
    assert response.json()["songs"][0]["title"] == "Song"


def test_get_unknown_cluster_returns_404():
    response = api_client().get("/clusters/missing")

    assert response.status_code == 404
