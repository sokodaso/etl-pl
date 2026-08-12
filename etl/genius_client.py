from __future__ import annotations

from typing import Any

import lyricsgenius as lg


def build_genius_client(access_token: str | None):
    if not access_token:
        raise ValueError("Set GENIUS_ACCESS_TOKEN or ACCESS_TOKEN before running Genius extraction.")

    genius = lg.Genius(access_token, timeout=15, retries=3)
    genius.remove_section_headers = True
    genius.skip_non_songs = True
    genius.excluded_terms = ["(Remix)", "(Live)"]
    return genius


def search_song_metadata(genius: Any, artist: str, title: str) -> dict[str, Any] | None:
    song = genius.search_song(title=title, artist=artist)
    if not song:
        return None

    data = song.to_dict()
    stats = data.get("stats") or {}

    return {
        "genius_song_id": data.get("id"),
        "genius_pageviews": stats.get("pageviews") or data.get("pageviews"),
        "genius_annotation_count": stats.get("accepted_annotations"),
        "genius_url": data.get("url"),
    }
