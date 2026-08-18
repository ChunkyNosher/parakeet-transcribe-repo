#!/usr/bin/env python3
"""A/B quality harness for the Parakeet Transcribe Docker app.

Drives the running Gradio app (inside the Docker container) over HTTP and
transcribes YouTube videos through the app's own YouTube pipeline, comparing
transcription quality and first-load timing across inference modes.

Modes
-----
ready      Ready-state snapshot restore (default; PARAKEET_READY_RESTORE=1).
standard   Standard restore path (PARAKEET_READY_RESTORE=0).
offline    Force offline long-form inference (PARAKEET_FORCE_INFERENCE_MODE=offline).
streaming  Force buffered streaming inference (PARAKEET_FORCE_INFERENCE_MODE=streaming).

Each mode is applied by restarting the container with a temporary Compose
override, so the switch is exactly what setting the env var in compose.yaml
would do. The container is restored to its default configuration when the run
finishes (unless --keep-mode is given).

Corpus
------
Pass URLs with --urls, or put one URL per line in a corpus file (default:
docker-data/ab-corpus/urls.txt) and pass --corpus. Include at least one video
shorter than ~30s and one longer, so both the offline and the buffered
streaming paths are exercised.

Examples
--------
python scripts/ab_quality.py --mode ready --mode standard --urls "https://youtu.be/..."
python scripts/ab_quality.py --mode offline --mode streaming --corpus docker-data/ab-corpus/urls.txt
python scripts/ab_quality.py --dry-run
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from gradio_client import Client

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = REPO_ROOT / "compose.yaml"
OVERRIDE_FILE = REPO_ROOT / "docker-data" / "ab-override.yaml"
RESULTS_DIR = REPO_ROOT / "docker-data" / "ab-results"
DEFAULT_CORPUS = REPO_ROOT / "docker-data" / "ab-corpus" / "urls.txt"
BASE_URL = "http://127.0.0.1:7860"
DIFF_LIMIT_LINES = 200

MODES = {
    "ready": {"PARAKEET_READY_RESTORE": "1"},
    "standard": {"PARAKEET_READY_RESTORE": "0"},
    "offline": {"PARAKEET_FORCE_INFERENCE_MODE": "offline"},
    "streaming": {"PARAKEET_FORCE_INFERENCE_MODE": "streaming"},
}


def _run_compose(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _container_running() -> bool:
    result = _run_compose(["ps", "--status", "running", "--format", "{{.Service}}"])
    return "parakeet-transcribe" in result.stdout


def _wait_http(url: str, timeout: float = 180.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5):
                return True
        except Exception:
            time.sleep(2)
    return False


def _write_override(env: dict[str, str]) -> None:
    lines = ["services:", "  parakeet-transcribe:", "    environment:"]
    for key, value in env.items():
        lines.append(f'      {key}: "{value}"')
    OVERRIDE_FILE.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDE_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _apply_mode(mode: str) -> None:
    _write_override(MODES[mode])
    result = _run_compose(["-f", str(COMPOSE_FILE), "-f", str(OVERRIDE_FILE), "up", "-d"])
    if result.returncode != 0:
        raise SystemExit(f"docker compose up failed for mode {mode}:\n{result.stderr}")
    if not _wait_http(BASE_URL):
        raise SystemExit(f"Container did not become healthy after switching to mode {mode}.")
    print(f"[ab] container restarted in mode '{mode}'", flush=True)


def _restore_defaults() -> None:
    result = _run_compose(["up", "-d"])
    if result.returncode != 0:
        print(f"[ab] WARNING: could not restore default container config:\n{result.stderr}", flush=True)
        return
    _wait_http(BASE_URL, timeout=180.0)
    print("[ab] container restored to default configuration", flush=True)


def _make_client() -> Client:
    return Client(
        BASE_URL,
        verbose=False,
        analytics_enabled=False,
        httpx_kwargs={"timeout": httpx.Timeout(timeout=None)},
    )


def _find_youtube_endpoint(client: Client) -> str:
    api = client.view_api(return_format="dict", print_info=False)
    named = api.get("named_endpoints", {}) or {}
    for name in named:
        if "youtube" in name.lower():
            return name
    raise SystemExit(
        "Could not find the YouTube transcription endpoint. Available endpoints: "
        + ", ".join(sorted(named))
    )


def _first_file_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            path = _first_file_path(item)
            if path is not None:
                return path
        return None
    path = Path(str(value))
    return path if path.is_file() else None


def _transcribe_youtube(client: Client, endpoint: str, url: str, model: str) -> dict[str, Any]:
    outputs = client.predict(
        url,
        model,
        "auto",
        2,
        "",
        1.0,
        False,
        False,
        False,
        False,
        api_name=endpoint,
    )
    status = str(outputs[0] or "")
    if status.startswith("###"):
        return {"error": status, "url": url}
    json_path = _first_file_path(outputs[5])
    if json_path is None:
        return {"error": "No JSON artifact returned.", "url": url, "status": status}
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    data["_status"] = status
    return data


def _video_id(url: str) -> str | None:
    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{6,})", url)
    return match.group(1) if match else None


def _video_id_from_source(source_name: str) -> str | None:
    match = re.search(r"\[([A-Za-z0-9_-]{6,})\]\s*$", source_name)
    return match.group(1) if match else None


def _capture_logs(mode: str, since: datetime) -> Path:
    result = _run_compose(["logs", "--since", since.isoformat()])
    log_path = RESULTS_DIR / mode / "container.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    return log_path


def _load_time_from_logs(log_path: Path) -> float | None:
    match = re.search(r"ready in ([\d.]+)s", log_path.read_text(encoding="utf-8", errors="replace"))
    return float(match.group(1)) if match else None


def _stats(data: dict[str, Any]) -> dict[str, Any]:
    words = data.get("words") or []
    confidences = [
        word.get("confidence")
        for word in words
        if isinstance(word.get("confidence"), (int, float))
    ]
    runtime = data.get("runtime") or {}
    return {
        "words": len(words),
        "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "segments": len(data.get("segments") or []),
        "duration_seconds": data.get("duration_seconds"),
        "inference_mode": runtime.get("inference_mode"),
        "elapsed_seconds": runtime.get("elapsed_seconds"),
        "warnings": data.get("warnings") or [],
    }


def _text_diff(baseline: str, other: str) -> str:
    diff = list(
        difflib.unified_diff(
            baseline.splitlines(),
            other.splitlines(),
            fromfile="baseline",
            tofile="other",
            lineterm="",
        )
    )
    if not diff:
        return "(identical)"
    if len(diff) > DIFF_LIMIT_LINES:
        diff = diff[:DIFF_LIMIT_LINES] + [f"... ({len(diff) - DIFF_LIMIT_LINES} more diff lines)"]
    return "\n".join(diff)


def _wer(reference: str, hypothesis: str) -> float:
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return 1.0 - difflib.SequenceMatcher(None, ref, hyp).ratio()


def _load_reference(reference_dir: Path, video_id: str) -> str | None:
    for candidate in (reference_dir / f"{video_id}.txt", reference_dir / f"{video_id.lower()}.txt"):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    return None


def _build_report(
    results: dict[str, dict[str, Any]],
    baseline: str,
    reference_dir: Path | None,
) -> str:
    lines: list[str] = []
    lines.append("Parakeet Transcribe A/B quality report")
    lines.append(f"baseline mode: {baseline}; compared modes: {', '.join(results)}")
    lines.append("")
    video_ids: list[str] = []
    for mode_results in results.values():
        for key in mode_results:
            if key != "_load_time" and key not in video_ids:
                video_ids.append(key)
    for video_id in video_ids:
        lines.append(f"=== {video_id} ===")
        lines.append(
            f"{'mode':<10} {'words':>6} {'conf':>6} {'segs':>5} {'dur_s':>7} "
            f"{'inference':<24} {'elapsed_s':>9} {'load_s':>6}"
        )
        for mode, mode_results in results.items():
            data = mode_results.get(video_id)
            if data is None:
                lines.append(f"{mode:<10} (no result)")
                continue
            if "error" in data:
                lines.append(f"{mode:<10} ERROR: {data['error']}")
                continue
            stats = _stats(data)
            load = mode_results.get("_load_time")
            lines.append(
                f"{mode:<10} {stats['words']:>6} {str(stats['mean_confidence']):>6} "
                f"{stats['segments']:>5} {str(stats['duration_seconds']):>7} "
                f"{str(stats['inference_mode']):<24} {str(stats['elapsed_seconds']):>9} "
                f"{str(load):>6}"
            )
            for warning in stats["warnings"]:
                lines.append(f"{'':<10} warning: {warning}")
        baseline_data = results[baseline].get(video_id)
        for mode, mode_results in results.items():
            if mode == baseline:
                continue
            other = mode_results.get(video_id)
            if (
                baseline_data is None
                or other is None
                or "error" in baseline_data
                or "error" in other
            ):
                continue
            lines.append("")
            lines.append(f"--- text diff ({mode} vs {baseline}) ---")
            lines.append(_text_diff(baseline_data.get("text", ""), other.get("text", "")))
        if reference_dir is not None:
            reference = _load_reference(reference_dir, video_id)
            if reference is not None:
                lines.append("")
                lines.append("--- WER vs reference ---")
                for mode, mode_results in results.items():
                    data = mode_results.get(video_id)
                    if data is not None and "error" not in data:
                        lines.append(f"{mode:<10} wer~={_wer(reference, data.get('text', '')):.4f}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B quality harness for the Parakeet Transcribe Docker app (YouTube corpus)."
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=sorted(MODES),
        default=None,
        help="Inference mode(s) to compare; repeatable. Default: ready standard.",
    )
    parser.add_argument("--urls", nargs="*", default=None, help="YouTube URLs to transcribe.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="File with one YouTube URL per line (# comments allowed). "
        f"Default: {DEFAULT_CORPUS} if it exists.",
    )
    parser.add_argument("--model", default="parakeet-v3", help="Model key (default: parakeet-v3).")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Directory with <video_id>.txt reference transcripts for WER.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Connect to the app, discover the YouTube endpoint, print the plan, and exit.",
    )
    parser.add_argument(
        "--keep-mode",
        action="store_true",
        help="Leave the container in the last mode instead of restoring defaults.",
    )
    args = parser.parse_args()

    modes = args.mode or ["ready", "standard"]
    urls: list[str] = []
    if args.urls:
        urls.extend(args.urls)
    corpus_file = args.corpus or DEFAULT_CORPUS
    if corpus_file.is_file():
        for line in corpus_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    if not urls:
        raise SystemExit(
            "No YouTube URLs. Pass --urls or create a corpus file at "
            f"{DEFAULT_CORPUS} (one URL per line)."
        )

    client = _make_client()
    endpoint = _find_youtube_endpoint(client)
    if args.dry_run:
        print(f"[ab] base URL: {BASE_URL}")
        print(f"[ab] YouTube endpoint: {endpoint}")
        print(f"[ab] modes: {', '.join(modes)}")
        print(f"[ab] corpus ({len(urls)} URLs):")
        for url in urls:
            print(f"      {url}")
        print("[ab] dry run: no container restart, no transcription.")
        return

    if not _container_running():
        raise SystemExit(
            "The transcription container is not running. Start it with `docker compose up -d` first."
        )

    baseline = modes[0]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    try:
        for mode in modes:
            _apply_mode(mode)
            since = datetime.now(UTC)
            client = _make_client()
            endpoint = _find_youtube_endpoint(client)
            mode_dir = RESULTS_DIR / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            mode_results: dict[str, Any] = {}
            for url in urls:
                print(f"[ab] [{mode}] transcribing {url}", flush=True)
                try:
                    data = _transcribe_youtube(client, endpoint, url, args.model)
                except Exception as exc:  # noqa: BLE001 - record any failure for the report
                    data = {"error": f"{type(exc).__name__}: {exc}", "url": url}
                video_id = _video_id(url)
                if video_id is None:
                    video_id = _video_id_from_source(str(data.get("source_name", "")))
                video_id = video_id or "video"
                safe_id = re.sub(r"[^A-Za-z0-9._-]+", "-", video_id)
                (mode_dir / f"{safe_id}.json").write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                mode_results[safe_id] = data
            log_path = _capture_logs(mode, since)
            mode_results["_load_time"] = _load_time_from_logs(log_path)
            results[mode] = mode_results
    finally:
        if not args.keep_mode:
            _restore_defaults()

    report = _build_report(results, baseline, args.reference_dir)
    report_path = RESULTS_DIR / f"report-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"[ab] report saved to {report_path}")


if __name__ == "__main__":
    sys.exit(main())
