from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .types import TranscriptionError


@dataclass(frozen=True)
class YouTubeAudio:
    path: Path
    title: str
    video_id: str
    webpage_url: str

    @property
    def source_name(self) -> str:
        return f"{self.title} [{self.video_id}]"


def validate_youtube_url(value: str) -> str:
    url = value.strip()
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    is_youtube_host = host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")
    is_youtube_host = is_youtube_host or host == "youtube-nocookie.com" or host.endswith(".youtube-nocookie.com")
    if parsed.scheme not in {"http", "https"} or not host or not is_youtube_host:
        raise TranscriptionError("Enter a valid YouTube video URL.")
    if not parsed.path or parsed.username or parsed.password:
        raise TranscriptionError("Enter a valid YouTube video URL.")
    return url


def _downloaded_path(info: dict[str, object], download_dir: Path, downloader: object) -> Path:
    candidates: list[Path] = []
    requested = info.get("requested_downloads")
    if isinstance(requested, list):
        for item in requested:
            if isinstance(item, dict):
                for key in ("filepath", "filename"):
                    value = item.get(key)
                    if isinstance(value, str):
                        candidates.append(Path(value))
    prepare_filename = getattr(downloader, "prepare_filename", None)
    if callable(prepare_filename):
        filename = prepare_filename(info)
        if isinstance(filename, str):
            candidates.append(Path(filename))
    for path in candidates:
        if path.is_file():
            return path
    files = [path for path in download_dir.iterdir() if path.is_file() and path.suffix != ".part"]
    if len(files) == 1:
        return files[0]
    raise TranscriptionError("YouTube did not produce a downloadable audio stream.")


def download_youtube_audio(url: str, download_dir: Path) -> YouTubeAudio:
    """Download one YouTube video's best audio stream without retaining a playlist."""
    validated_url = validate_youtube_url(url)
    download_dir.mkdir(parents=True, exist_ok=True)
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:  # pragma: no cover - installation is locked in production.
        raise TranscriptionError("YouTube support is unavailable because yt-dlp is not installed.") from exc

    options = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "outtmpl": str(download_dir / "%(id)s.%(ext)s"),
        "restrictfilenames": True,
        "quiet": True,
        "no_warnings": True,
        "overwrites": True,
    }
    try:
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(validated_url, download=True)
            if not isinstance(info, dict):
                raise TranscriptionError("YouTube did not return video information for this URL.")
            path = _downloaded_path(info, download_dir, downloader)
    except TranscriptionError:
        raise
    except Exception as exc:
        raise TranscriptionError(f"Could not download YouTube audio: {exc}") from exc

    title = str(info.get("title") or path.stem)
    video_id = str(info.get("id") or path.stem)
    webpage_url = str(info.get("webpage_url") or validated_url)
    return YouTubeAudio(path=path, title=title, video_id=video_id, webpage_url=webpage_url)
