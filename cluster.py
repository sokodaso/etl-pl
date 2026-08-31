from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from etl.config import load_settings
from etl.loader import prepare_for_sql

NUMERIC_FEATURES = [
    "rank", "last_rank", "peak_rank", "weeks", "youtube_views", "youtube_likes",
    "youtube_comments", "genius_pageviews", "genius_annotation_count",
    "yt_like_to_view_ratio", "yt_like_to_comment_ratio",
]
CATEGORICAL_FEATURES = ["is_new"]
COUNT_FEATURES = [
    "youtube_views", "youtube_likes", "youtube_comments", "genius_pageviews",
    "genius_annotation_count",
]
IDENTITY_COLUMNS = ["chart_date", "title", "artist"]


@dataclass
class ClusteringResult:
    run_id: str
    created_at: datetime
    parameters: dict[str, Any]
    metrics: dict[str, Any]
    assignments: pd.DataFrame
    profiles: pd.DataFrame
    pipeline: dict[str, Any]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
    ])
    return ColumnTransformer([
        ("numerical", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    likes = pd.to_numeric(result["youtube_likes"], errors="coerce")
    result["yt_like_to_view_ratio"] = likes / pd.to_numeric(result["youtube_views"], errors="coerce")
    result["yt_like_to_comment_ratio"] = likes / pd.to_numeric(result["youtube_comments"], errors="coerce")
    return result


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    required = IDENTITY_COLUMNS + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Clustering dataset is missing columns: {', '.join(missing)}")
    features = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for column in COUNT_FEATURES:
        values = pd.to_numeric(features[column], errors="coerce").clip(lower=0)
        features[column] = np.log1p(values)
    features["is_new"] = pd.to_numeric(features["is_new"], errors="coerce").astype(float)
    return features.replace([np.inf, -np.inf], np.nan)


def evaluate_labels(points: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    mask = labels != -1
    cluster_labels = set(labels[mask])
    metrics: dict[str, Any] = {
        "cluster_count": len(cluster_labels),
        "noise_count": int((~mask).sum()),
        "noise_ratio": float((~mask).mean()),
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }
    if len(cluster_labels) >= 2 and int(mask.sum()) > len(cluster_labels):
        metrics.update({
            "silhouette": float(silhouette_score(points[mask], labels[mask])),
            "davies_bouldin": float(davies_bouldin_score(points[mask], labels[mask])),
            "calinski_harabasz": float(calinski_harabasz_score(points[mask], labels[mask])),
        })
    return metrics


def select_dbscan(
    points: np.ndarray,
    eps_values: Iterable[float],
    min_samples_values: Iterable[int],
) -> tuple[DBSCAN, dict[str, Any], pd.DataFrame]:
    candidates: list[dict[str, Any]] = []
    best: tuple[float, int] | None = None
    best_score = -np.inf
    for min_samples in min_samples_values:
        for eps in eps_values:
            labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(points)
            metrics = evaluate_labels(points, labels)
            candidates.append({"eps": float(eps), "min_samples": int(min_samples), **metrics})
            score = metrics["silhouette"]
            if score is not None and score > best_score:
                best = (float(eps), int(min_samples))
                best_score = score
    if best is None:
        raise ValueError("No DBSCAN candidate produced at least two non-noise clusters.")
    model = DBSCAN(eps=best[0], min_samples=best[1]).fit(points)
    return model, {"eps": best[0], "min_samples": best[1]}, pd.DataFrame(candidates)


def _song_key(row: pd.Series) -> str:
    identity = "|".join(str(row[column]) for column in IDENTITY_COLUMNS)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _json_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, allow_nan=False)


def _feature_summary(rows: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(rows[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not values.empty:
            summary[column] = {
                "mean": float(values.mean()), "median": float(values.median()),
                "min": float(values.min()), "max": float(values.max()),
            }
    is_new = rows["is_new"].dropna().astype(bool)
    summary["is_new_ratio"] = float(is_new.mean()) if not is_new.empty else None
    return summary


def match_stable_groups(
    cluster_song_keys: dict[int, set[str]],
    previous_assignments: pd.DataFrame | None,
    minimum_overlap: float,
) -> dict[int, str]:
    matches: dict[int, str] = {}
    if previous_assignments is not None and not previous_assignments.empty:
        candidates: list[tuple[float, int, str]] = []
        previous_groups = previous_assignments.dropna(subset=["stable_group_id"]).groupby("stable_group_id")
        for label, current_keys in cluster_song_keys.items():
            for group_id, rows in previous_groups:
                previous_keys = set(rows["song_key"])
                union = current_keys | previous_keys
                overlap = len(current_keys & previous_keys) / len(union) if union else 0.0
                candidates.append((overlap, label, group_id))
        used_labels: set[int] = set()
        used_groups: set[str] = set()
        for overlap, label, group_id in sorted(candidates, reverse=True):
            if overlap >= minimum_overlap and label not in used_labels and group_id not in used_groups:
                matches[label] = group_id
                used_labels.add(label)
                used_groups.add(group_id)
    for label in cluster_song_keys:
        matches.setdefault(label, f"group_{uuid.uuid4().hex[:12]}")
    return matches


def build_profiles_and_assignments(
    source_df: pd.DataFrame,
    points: np.ndarray,
    labels: np.ndarray,
    run_id: str,
    previous_assignments: pd.DataFrame | None = None,
    minimum_group_overlap: float = 0.2,
    representative_count: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    centroids = {
        int(label): points[labels == label].mean(axis=0)
        for label in sorted(set(labels)) if label != -1
    }
    assignments: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    source_df = source_df.reset_index(drop=True)
    song_keys = [_song_key(row) for _, row in source_df.iterrows()]
    cluster_song_keys = {
        label: {song_keys[position] for position in np.flatnonzero(labels == label)}
        for label in centroids
    }
    stable_groups = match_stable_groups(cluster_song_keys, previous_assignments, minimum_group_overlap)
    for position, row in source_df.iterrows():
        label = int(labels[position])
        assignments.append({
            "run_id": run_id, "song_key": song_keys[position], "chart_date": row["chart_date"],
            "title": row["title"], "artist": row["artist"],
            "dbscan_label": None if label == -1 else label,
            "stable_group_id": stable_groups.get(label), "is_noise": label == -1,
        })
    for label, centroid in centroids.items():
        positions = np.flatnonzero(labels == label)
        distances = np.linalg.norm(points[positions] - centroid, axis=1)
        representative_positions = positions[np.argsort(distances)[:representative_count]]
        representatives = [{
            "song_key": assignments[position]["song_key"],
            "title": assignments[position]["title"], "artist": assignments[position]["artist"],
            "chart_date": assignments[position]["chart_date"],
            "distance_to_centroid": float(np.linalg.norm(points[position] - centroid)),
        } for position in representative_positions]
        profiles.append({
            "run_id": run_id, "stable_group_id": stable_groups[label], "dbscan_label": label,
            "cluster_size": len(positions), "centroid_pca_json": _json_value(centroid.tolist()),
            "feature_summary_json": _json_value(_feature_summary(source_df.iloc[positions])),
            "representative_songs_json": _json_value(representatives),
        })
    return pd.DataFrame(assignments), pd.DataFrame(profiles)


def fit_clustering(
    source_df: pd.DataFrame,
    previous_assignments: pd.DataFrame | None = None,
    *,
    n_components: int = 6,
    eps_values: Iterable[float] = tuple(np.arange(0.5, 2.5, 0.05)),
    min_samples_values: Iterable[int] = (5,),
    minimum_group_overlap: float = 0.2,
) -> ClusteringResult:
    if len(source_df) < 3:
        raise ValueError("At least three rows are required for clustering.")
    duplicate_rows = source_df.duplicated(IDENTITY_COLUMNS, keep=False)
    if duplicate_rows.any():
        raise ValueError(
            "Clustering dataset contains duplicate chart_date/title/artist rows; "
            "deduplicate the source snapshot before fitting."
        )
    source_df = add_ratio_features(source_df).reset_index(drop=True)
    features = prepare_features(source_df)
    preprocessor = build_preprocessor()
    processed = preprocessor.fit_transform(features)
    component_count = min(n_components, processed.shape[0], processed.shape[1])
    pca = PCA(n_components=component_count)
    points = pca.fit_transform(processed)
    dbscan, parameters, search_results = select_dbscan(points, eps_values, min_samples_values)
    labels = dbscan.labels_
    run_id = uuid.uuid4().hex
    metrics = {
        **evaluate_labels(points, labels), "dataset_size": len(source_df),
        "pca_components": component_count,
        "pca_explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
    }
    assignments, profiles = build_profiles_and_assignments(
        source_df, points, labels, run_id, previous_assignments, minimum_group_overlap
    )
    pipeline = {
        "preprocessor": preprocessor, "pca": pca, "dbscan": dbscan,
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "search_results": search_results,
    }
    return ClusteringResult(
        run_id, datetime.now(timezone.utc), parameters, metrics, assignments, profiles, pipeline
    )


def ensure_clustering_tables(engine: Any) -> None:
    from sqlalchemy import Boolean, Column, Date, DateTime, Float, ForeignKey, Integer, MetaData, String, Table, Text
    metadata = MetaData()
    Table("clustering_runs", metadata,
        Column("run_id", String(32), primary_key=True), Column("created_at", DateTime(timezone=True), nullable=False),
        Column("status", String(20), nullable=False, index=True), Column("dataset_size", Integer, nullable=False),
        Column("cluster_count", Integer, nullable=False), Column("noise_count", Integer, nullable=False),
        Column("noise_ratio", Float, nullable=False), Column("parameters_json", Text, nullable=False),
        Column("metrics_json", Text, nullable=False), Column("artifact_path", Text, nullable=False))
    Table("music_groups", metadata,
        Column("stable_group_id", String(32), primary_key=True), Column("created_run_id", String(32), nullable=False),
        Column("last_seen_run_id", String(32), nullable=False), Column("status", String(20), nullable=False))
    Table("cluster_profiles", metadata,
        Column("run_id", String(32), ForeignKey("clustering_runs.run_id"), primary_key=True),
        Column("stable_group_id", String(32), ForeignKey("music_groups.stable_group_id"), primary_key=True),
        Column("dbscan_label", Integer, nullable=False), Column("cluster_size", Integer, nullable=False),
        Column("centroid_pca_json", Text, nullable=False), Column("feature_summary_json", Text, nullable=False),
        Column("representative_songs_json", Text, nullable=False))
    Table("song_cluster_assignments", metadata,
        Column("run_id", String(32), ForeignKey("clustering_runs.run_id"), primary_key=True),
        Column("song_key", String(64), primary_key=True), Column("chart_date", Date),
        Column("title", Text, nullable=False), Column("artist", Text, nullable=False),
        Column("dbscan_label", Integer), Column("stable_group_id", String(32), ForeignKey("music_groups.stable_group_id")),
        Column("is_noise", Boolean, nullable=False, index=True))
    metadata.create_all(engine)


def load_active_profiles(engine: Any) -> pd.DataFrame:
    return pd.read_sql("""
        SELECT cp.* FROM cluster_profiles cp
        JOIN clustering_runs cr ON cr.run_id = cp.run_id
        WHERE cr.status = 'active'
    """, engine)


def load_active_assignments(engine: Any) -> pd.DataFrame:
    return pd.read_sql("""
        SELECT sca.* FROM song_cluster_assignments sca
        JOIN clustering_runs cr ON cr.run_id = sca.run_id
        WHERE cr.status = 'active'
    """, engine)


def persist_result(result: ClusteringResult, engine: Any, artifact_dir: Path) -> Path:
    from sqlalchemy import MetaData, Table, insert, select, update
    ensure_clustering_tables(engine)
    artifact_dir = artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"clustering_{result.run_id}.joblib"
    temporary_path = artifact_path.with_suffix(".tmp")
    joblib.dump(result.pipeline, temporary_path)
    temporary_path.replace(artifact_path)
    metadata = MetaData()
    table_names = ["clustering_runs", "music_groups", "cluster_profiles", "song_cluster_assignments"]
    metadata.reflect(bind=engine, only=table_names)
    runs, groups, profiles, assignments = (Table(name, metadata) for name in table_names)
    try:
        with engine.begin() as connection:
            connection.execute(update(runs).where(runs.c.status == "active").values(status="superseded"))
            connection.execute(insert(runs).values(
                run_id=result.run_id, created_at=result.created_at, status="active",
                dataset_size=result.metrics["dataset_size"], cluster_count=result.metrics["cluster_count"],
                noise_count=result.metrics["noise_count"], noise_ratio=result.metrics["noise_ratio"],
                parameters_json=_json_value(result.parameters), metrics_json=_json_value(result.metrics),
                artifact_path=str(artifact_path)))
            for profile in result.profiles.to_dict("records"):
                group_id = profile["stable_group_id"]
                existing = connection.execute(select(groups.c.stable_group_id).where(groups.c.stable_group_id == group_id)).first()
                if existing:
                    connection.execute(update(groups).where(groups.c.stable_group_id == group_id).values(
                        last_seen_run_id=result.run_id, status="active"))
                else:
                    connection.execute(insert(groups).values(
                        stable_group_id=group_id, created_run_id=result.run_id,
                        last_seen_run_id=result.run_id, status="active"))
            active_group_ids = result.profiles["stable_group_id"].tolist()
            if active_group_ids:
                connection.execute(update(groups).where(~groups.c.stable_group_id.in_(active_group_ids)).values(status="inactive"))
            profile_records = prepare_for_sql(result.profiles).to_dict("records")
            assignment_records = prepare_for_sql(result.assignments).to_dict("records")
            connection.execute(insert(profiles), profile_records)
            connection.execute(insert(assignments), assignment_records)
    except Exception:
        artifact_path.unlink(missing_ok=True)
        raise
    return artifact_path


def load_source_data(engine: Any, table_name: str = "song_week_stats") -> pd.DataFrame:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", table_name):
        raise ValueError("Invalid source table name.")
    columns = IDENTITY_COLUMNS + [
        "rank", "last_rank", "peak_rank", "weeks", "is_new", "youtube_views",
        "youtube_likes", "youtube_comments", "genius_pageviews", "genius_annotation_count",
    ]
    quote = engine.dialect.identifier_preparer.quote_identifier
    qualified_table = ".".join(quote(part) for part in table_name.split("."))
    selected_columns = ", ".join(quote(column) for column in columns)
    return pd.read_sql(f"SELECT {selected_columns} FROM {qualified_table}", engine)


def run_clustering(mysql_url: str, source_table: str, artifact_dir: Path) -> ClusteringResult:
    from sqlalchemy import create_engine
    engine = create_engine(mysql_url)
    ensure_clustering_tables(engine)
    result = fit_clustering(load_source_data(engine, source_table), load_active_assignments(engine))
    persist_result(result, engine, artifact_dir)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and persist versioned DBSCAN music groups.")
    parser.add_argument("--source-table", default="song_week_stats")
    parser.add_argument("--artifact-dir", default="artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    if not settings.mysql_url:
        raise ValueError("Set MYSQL_URL before running clustering.")
    result = run_clustering(settings.mysql_url, args.source_table, Path(args.artifact_dir))
    print(json.dumps({"run_id": result.run_id, "parameters": result.parameters, "metrics": result.metrics}, indent=2))


if __name__ == "__main__":
    main()
