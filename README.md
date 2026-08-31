# Billboard Hot 100 Music Metrics ETL

This project builds a weekly `song_week_stats` dataset, groups songs by their
chart and engagement characteristics, and exposes the active groups through an
HTTP API. The ETL dataset combines:

1. Billboard Hot 100 chart rows
2. YouTube search/video statistics for each charting song
3. Genius song metadata for each charting song
4. A MySQL load step for the final normalized table

The intended grain is one row per song per Billboard chart week. The clustering
stage uses normalized chart, YouTube, and Genius features with PCA and DBSCAN.
Each successful run is versioned in MySQL, saved as a Joblib artifact, and made
active for API reads while the previous run is retained as superseded.

## Output Columns

The final dataframe and MySQL table include:

- `chart_date`
- `rank`
- `title`
- `artist`
- `last_rank`
- `peak_rank`
- `weeks`
- `is_new`
- `youtube_video_id`
- `youtube_title`
- `youtube_channel`
- `youtube_published_at`
- `youtube_views`
- `youtube_likes`
- `youtube_comments`
- `youtube_search_query`
- `genius_song_id`
- `genius_pageviews`
- `genius_annotation_count`
- `genius_url`

## Environment Variables

Install dependencies:

```bash
pip install -r requirements.txt
```

Required for YouTube enrichment:

```bash
export YOUTUBE_API_KEY="your-youtube-data-api-key"
```

If you do not have an API key, the script can fall back to OAuth:

```bash
export CLIENT_SECRET_PATH="/path/to/client_secret.json"
```

Required for Genius enrichment:

```bash
export GENIUS_ACCESS_TOKEN="your-genius-access-token"
```

Required for MySQL loading:

```bash
export MYSQL_URL="mysql+pymysql://user:password@localhost:3306/database_name"
```

Optional:

```bash
export BILLBOARD_CHART="hot-100"
export OUTPUT_DIR="output"
```

## Run It

For a small API test without loading MySQL:

```bash
python etl_pipeline.py --limit 5 --skip-load
```

For a full run and MySQL load:

```bash
python etl_pipeline.py
```

For chart-only development:

```bash
python etl_pipeline.py --limit 10 --skip-youtube --skip-genius --skip-load
```

## Build Music Groups

Clustering reads `song_week_stats` from MySQL, tunes and fits DBSCAN, preserves
stable group IDs when groups overlap a previous run, and writes run metadata,
song assignments, group profiles, and the fitted pipeline back to storage:

```bash
python cluster.py
```

Use a different source table or artifact directory when needed:

```bash
python cluster.py --source-table analytics.song_week_stats --artifact-dir artifacts
```

This step requires `MYSQL_URL` and at least three source rows. Its default
artifact directory is `artifacts/`.

## Cluster API

After a clustering result has been persisted, start the API with:

```bash
uvicorn api:app --reload
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.
The API reads only the active clustering run and provides:

- `GET /clusters`: group IDs, DBSCAN labels, sizes, feature summaries, and
  representative songs, ordered by group size
- `GET /clusters/{group_id}`: the same profile plus every assigned song; an
  unknown group returns `404`

Example:

```bash
curl http://127.0.0.1:8000/clusters
curl http://127.0.0.1:8000/clusters/group_abc123
```

## Tests

Run the automated ETL, clustering, persistence, and API tests with:

```bash
python -m pytest
```

## Files

- `etl_pipeline.py`: orchestration entrypoint
- `etl/billboard_client.py`: Billboard chart extraction
- `etl/youtube_client.py`: YouTube search and video statistics
- `etl/genius_client.py`: Genius song metadata
- `etl/transform.py`: final `song_week_stats` dataframe assembly
- `etl/loader.py`: MySQL load via pandas and SQLAlchemy
- `cluster.py`: PCA/DBSCAN fitting, evaluation, stable group matching, and persistence
- `api.py`: FastAPI endpoints for active cluster profiles and song assignments
- `artifacts/`: persisted clustering pipeline files
- `tests/`: ETL, clustering, persistence, and API tests
