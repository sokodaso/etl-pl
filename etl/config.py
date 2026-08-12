from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    chart_name: str
    output_dir: str
    youtube_api_key: str | None
    youtube_client_secret_path: str | None
    genius_access_token: str | None
    mysql_url: str | None


def load_settings() -> Settings:
    return Settings(
        chart_name=os.getenv("BILLBOARD_CHART", "hot-100"),
        output_dir=os.getenv("OUTPUT_DIR", "output"),
        youtube_api_key=os.getenv("YOUTUBE_API_KEY"),
        youtube_client_secret_path=os.getenv("CLIENT_SECRET_PATH"),
        genius_access_token=os.getenv("GENIUS_ACCESS_TOKEN") or os.getenv("ACCESS_TOKEN"),
        mysql_url=os.getenv("MYSQL_URL"),
    )
