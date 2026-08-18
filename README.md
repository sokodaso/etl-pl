# Billboard Hot 100 Music Metrics ETL

This project builds a weekly `song_week_stats` dataset from:

1. Billboard Hot 100 chart rows
2. YouTube search/video statistics for each charting song
3. Genius song metadata for each charting song
4. A MySQL load step for the final normalized table

The intended grain is one row per song per Billboard chart week.

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

## Files

- `etl_pipeline.py`: orchestration entrypoint
- `etl/billboard_client.py`: Billboard chart extraction
- `etl/youtube_client.py`: YouTube search and video statistics
- `etl/genius_client.py`: Genius song metadata
- `etl/transform.py`: final `song_week_stats` dataframe assembly
- `etl/loader.py`: MySQL load via pandas and SQLAlchemy
