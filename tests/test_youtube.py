import sys
from types import SimpleNamespace

import pytest

from parakeet_transcribe.types import TranscriptionError
from parakeet_transcribe.youtube import download_youtube_audio, validate_youtube_url


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=abc123",
        "https://youtu.be/abc123?t=12",
        "https://music.youtube.com/watch?v=abc123",
        "https://www.youtube-nocookie.com/embed/abc123",
    ],
)
def test_validate_youtube_url_accepts_youtube_hosts(url) -> None:
    assert validate_youtube_url(url) == url


@pytest.mark.parametrize(
    "url",
    ["https://example.com/video", "ftp://youtube.com/watch?v=abc", "https://evil-youtube.com/watch?v=abc"],
)
def test_validate_youtube_url_rejects_non_youtube_hosts(url) -> None:
    with pytest.raises(TranscriptionError, match="valid YouTube"):
        validate_youtube_url(url)


def test_download_youtube_audio_uses_single_best_audio_stream(tmp_path, monkeypatch) -> None:
    class FakeYoutubeDL:
        def __init__(self, options):
            assert options["format"] == "bestaudio/best"
            assert options["noplaylist"] is True
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def extract_info(self, url, *, download):
            path = tmp_path / "abc123.webm"
            path.write_bytes(b"audio")
            return {
                "id": "abc123",
                "title": "Example video",
                "webpage_url": url,
                "requested_downloads": [{"filepath": str(path)}],
            }

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))
    audio = download_youtube_audio("https://www.youtube.com/watch?v=abc123", tmp_path)
    assert audio.path.name == "abc123.webm"
    assert audio.source_name == "Example video [abc123]"
