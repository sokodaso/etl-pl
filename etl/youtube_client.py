from __future__ import annotations

import os
from typing import Any

import google_auth_oauthlib.flow
import googleapiclient.discovery


YOUTUBE_READONLY_SCOPE = ["https://www.googleapis.com/auth/youtube.readonly"]


def build_youtube_client(
    api_key: str | None = None,
    client_secret_path: str | None = None,
):
    if api_key:
        return googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

    if not client_secret_path:
        raise ValueError("Set YOUTUBE_API_KEY or CLIENT_SECRET_PATH before running YouTube extraction.")

    os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        client_secret_path,
        YOUTUBE_READONLY_SCOPE,
    )
    credentials = flow.run_local_server(port=0)
    return googleapiclient.discovery.build("youtube", "v3", credentials=credentials)


def search_top_video(youtube: Any, artist: str, title: str) -> dict[str, Any] | None:
    
    '''
        This function searches for the top YouTube video for a given artist and title.
        It returns a dictionary containing metadata about the video, including its ID, title, channel, publication date, view count, like count, comment count, and the search query used.
        If no video is found, it returns None.

    '''

    query = f"{artist} {title} official"
    search_response = (
        youtube.search()
        .list(
            part="snippet",
            q=query,
            type="video",
            maxResults=1,
            order="relevance",
            safeSearch="none",
        )
        .execute()
    )

    items = search_response.get("items", [])
    if not items:
        return None

    search_item = items[0]
    video_id = search_item["id"]["videoId"]
    video_response = (
        youtube.videos()
        .list(
            part="snippet,statistics",
            id=video_id,
        )
        .execute()
    )

    video_items = video_response.get("items", [])
    if not video_items:
        return None

    video = video_items[0]
    snippet = video.get("snippet", {})
    statistics = video.get("statistics", {})

    return {
        "youtube_video_id": video_id,
        "youtube_title": snippet.get("title"),
        "youtube_channel": snippet.get("channelTitle"),
        "youtube_published_at": snippet.get("publishedAt"),
        "youtube_views": _to_int(statistics.get("viewCount")),
        "youtube_likes": _to_int(statistics.get("likeCount")),
        "youtube_comments": _to_int(statistics.get("commentCount")),
        "youtube_search_query": query,
    }


def _to_int(value: str | int | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
