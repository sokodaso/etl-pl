from datetime import date, datetime, timezone
import json

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from cluster import (
    ClusteringResult,
    add_ratio_features,
    build_preprocessor,
    build_profiles_and_assignments,
    ensure_clustering_tables,
    match_stable_groups,
    persist_result,
    prepare_features,
)


def sample_rows() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "chart_date": date(2026, 8, 1),
            "title": f"Song {index}",
            "artist": f"Artist {index}",
            "rank": index + 1,
            "last_rank": index + 2,
            "peak_rank": 1,
            "weeks": index + 1,
            "is_new": index == 0,
            "youtube_views": 1000 * (index + 1),
            "youtube_likes": 100 * (index + 1),
            "youtube_comments": 10 * (index + 1),
            "genius_pageviews": 500 * (index + 1),
            "genius_annotation_count": index + 1,
            "yt_like_to_view_ratio": 0.1,
            "yt_like_to_comment_ratio": 10.0,
        }
        for index in range(5)
    ])


def test_profiles_keep_noise_separate_and_select_representatives():
    rows = sample_rows()
    points = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0], [20.0, 20.0]])
    labels = np.array([0, 0, 1, 1, -1])

    assignments, profiles = build_profiles_and_assignments(rows, points, labels, "run-one")

    assert assignments["is_noise"].sum() == 1
    assert assignments.loc[assignments["is_noise"], "stable_group_id"].isna().all()
    assert sorted(profiles["cluster_size"].tolist()) == [2, 2]
    representatives = json.loads(profiles.iloc[0]["representative_songs_json"])
    assert representatives[0]["title"].startswith("Song")


def test_stable_group_matching_uses_song_overlap():
    previous = pd.DataFrame([
        {"song_key": "a", "stable_group_id": "group_previous"},
        {"song_key": "b", "stable_group_id": "group_previous"},
    ])

    matches = match_stable_groups({7: {"a", "b", "c"}}, previous, minimum_overlap=0.2)

    assert matches[7] == "group_previous"


def test_preprocessor_accepts_boolean_is_new_values():
    features = prepare_features(add_ratio_features(sample_rows()))

    transformed = build_preprocessor().fit_transform(features)

    assert transformed.shape[0] == len(features)
    assert np.isfinite(transformed).all()


def test_persist_result_promotes_new_run_and_writes_all_views(tmp_path):
    engine = create_engine("sqlite://")
    ensure_clustering_tables(engine)
    rows = sample_rows()
    points = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0], [20.0, 20.0]])
    labels = np.array([0, 0, 1, 1, -1])
    assignments, profiles = build_profiles_and_assignments(rows, points, labels, "run-one")
    result = ClusteringResult(
        run_id="run-one",
        created_at=datetime.now(timezone.utc),
        parameters={"eps": 0.5, "min_samples": 2},
        metrics={"dataset_size": 5, "cluster_count": 2, "noise_count": 1, "noise_ratio": 0.2},
        assignments=assignments,
        profiles=profiles,
        pipeline={"name": "test-pipeline"},
    )

    artifact_path = persist_result(result, engine, tmp_path)

    assert artifact_path.exists()
    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM clustering_runs")).scalar() == 1
        assert connection.execute(text("SELECT COUNT(*) FROM cluster_profiles")).scalar() == 2
        assert connection.execute(text("SELECT COUNT(*) FROM song_cluster_assignments")).scalar() == 5
        assert connection.execute(text("SELECT status FROM clustering_runs")).scalar() == "active"
        noise_row = connection.execute(text("""
            SELECT dbscan_label, stable_group_id
            FROM song_cluster_assignments
            WHERE is_noise = 1
        """)).one()
        assert noise_row.dbscan_label is None
        assert noise_row.stable_group_id is None
