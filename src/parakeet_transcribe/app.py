from __future__ import annotations

from threading import Event

import gradio as gr

from .diagnostics import doctor_report
from .exports import create_run_directory, readable_summary, write_bundle, write_result
from .models import DEFAULT_MODEL_KEY, MODELS, get_model
from .service import MAX_BATCH_SIZE, TranscriptionService
from .types import CancelledError, TranscriptionError

SERVICE = TranscriptionService()
CANCEL_REQUESTED = Event()


def _model_choices() -> list[tuple[str, str]]:
    return [(spec.label, key) for key, spec in MODELS.items()]


def _model_details(model_key: str) -> str:
    spec = get_model(model_key)
    timestamp = (
        "Yes — TXT, JSON, CSV, SRT, and VTT"
        if spec.capabilities.timestamps
        else "No — TXT, JSON, and CSV only"
    )
    return (
        f"**{spec.model_id}**  \n"
        f"Languages: {spec.capabilities.supported_languages}; automatic detection: "
        f"{'yes' if spec.capabilities.automatic_language_detection else 'no'}  \n"
        f"Timestamp exports: {timestamp}"
    )


def _system_details() -> str:
    _, report = doctor_report()
    return report


def _unload() -> str:
    return SERVICE.unload()


def _request_cancel() -> str:
    CANCEL_REQUESTED.set()
    return "Cancellation requested; the active job will stop between chunks."


def _publish_results(results, run_dir) -> tuple[str, str, str | None, str | None, str | None, str | None]:
    artifacts = [write_result(result, run_dir) for result in results]
    bundle = write_bundle(results, run_dir)
    transcript = "\n\n".join(f"## {result.source_name}\n\n{result.text}" for result in results)
    status_parts = [f"- {readable_summary(result)}" for result in results]
    for result in results:
        if result.summary:
            status_parts.append(f"- Summary ready for {result.source_name}")
        if any(segment.speaker for segment in result.segments):
            status_parts.append(f"- Speaker labels attached for {result.source_name}")
    status = "\n".join(status_parts)
    first = artifacts[0]
    return (
        status,
        transcript,
        str(bundle),
        str(first["json"]),
        str(first.get("srt")) if first.get("srt") else None,
        str(first.get("vtt")) if first.get("vtt") else None,
    )


def _run_files(
    files: list[str] | None,
    model_key: str,
    language: str,
    batch_size: int,
    diarize: bool,
    summarize: bool,
    redact_pii: bool,
    clean_format: bool,
    progress: gr.Progress = gr.Progress(),  # noqa: B008 - Gradio injects this callback.
) -> tuple[str, str, str | None, str | None, str | None, str | None]:
    try:
        CANCEL_REQUESTED.clear()
        run_dir = create_run_directory()
        results = SERVICE.transcribe_files(
            files or [],
            model_key=model_key,
            language=language.strip() or "auto",
            batch_size=int(batch_size),
            work_dir=run_dir,
            progress=lambda fraction, description: progress(fraction, desc=description),
            cancel=CANCEL_REQUESTED.is_set,
            diarize=bool(diarize),
            summarize=bool(summarize),
            redact_pii=bool(redact_pii),
            clean_format=bool(clean_format),
        )
        return _publish_results(results, run_dir)
    except CancelledError as exc:
        return str(exc), "", None, None, None, None
    except TranscriptionError as exc:
        return f"### Transcription failed\n\n{exc}", "", None, None, None, None
    except Exception as exc:  # pragma: no cover - defensive UI boundary
        return f"### Unexpected failure\n\n`{type(exc).__name__}: {exc}`", "", None, None, None, None


def _run_youtube(
    url: str,
    model_key: str,
    language: str,
    batch_size: int,
    diarize: bool,
    summarize: bool,
    redact_pii: bool,
    clean_format: bool,
    progress: gr.Progress = gr.Progress(),  # noqa: B008 - Gradio injects this callback.
) -> tuple[str, str, str | None, str | None, str | None, str | None]:
    try:
        CANCEL_REQUESTED.clear()
        run_dir = create_run_directory()
        results = SERVICE.transcribe_youtube(
            url,
            model_key=model_key,
            language=language.strip() or "auto",
            batch_size=int(batch_size),
            work_dir=run_dir,
            progress=lambda fraction, description: progress(fraction, desc=description),
            cancel=CANCEL_REQUESTED.is_set,
            diarize=bool(diarize),
            summarize=bool(summarize),
            redact_pii=bool(redact_pii),
            clean_format=bool(clean_format),
        )
        return _publish_results(results, run_dir)
    except CancelledError as exc:
        return str(exc), "", None, None, None, None
    except TranscriptionError as exc:
        return f"### Transcription failed\n\n{exc}", "", None, None, None, None
    except Exception as exc:  # pragma: no cover - defensive UI boundary
        return f"### Unexpected failure\n\n`{type(exc).__name__}: {exc}`", "", None, None, None, None


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Parakeet Transcribe") as app:
        gr.Markdown(
            "# Parakeet Transcribe\n"
            "Local Windows file transcription with NVIDIA ASR checkpoints. Models download once and then run from the local cache."
        )
        with gr.Accordion("System diagnostics", open=False):
            diagnostics = gr.Markdown(_system_details())
            gr.Button("Refresh diagnostics", size="sm").click(_system_details, outputs=diagnostics)
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(
                    label="Audio or video files (any FFmpeg-readable format)",
                    file_count="multiple",
                    type="filepath",
                )
                gr.Markdown(
                    "FFmpeg validates and decodes uploads, including M4A, MP3, WAV, FLAC, MP4, MKV, and WebM."
                )
                youtube_url = gr.Textbox(
                    label="YouTube video URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    info="Downloads the best available audio stream from one YouTube video, then transcribes it locally.",
                )
                model = gr.Dropdown(_model_choices(), value=DEFAULT_MODEL_KEY, label="NVIDIA model")
                details = gr.Markdown(_model_details(DEFAULT_MODEL_KEY))
                model.change(_model_details, inputs=model, outputs=details)
                language = gr.Textbox(
                    value="auto",
                    label="Language",
                    info="Use auto, or a locale such as en-US or de-DE for Nemotron.",
                )
                batch_size = gr.Slider(
                    1,
                    MAX_BATCH_SIZE,
                    value=2,
                    step=1,
                    label="Chunk batch size",
                    info=f"Higher uses more VRAM (max {MAX_BATCH_SIZE}). Leave the model loaded between files for best throughput.",
                )
                diarize = gr.Checkbox(
                    label="Speaker diarization (local MFCC clustering)",
                    value=False,
                    info="Optional CPU post-pass. Best with Parakeet timestamps; quality is below commercial diarization APIs.",
                )
                summarize = gr.Checkbox(
                    label="Extractive summary + chapters",
                    value=False,
                    info="Local text-only chapters from pause gaps; no cloud LLM.",
                )
                redact_pii = gr.Checkbox(label="Redact PII in text exports", value=False)
                clean_format = gr.Checkbox(label="Light clean / smart formatting", value=False)
                with gr.Row():
                    run = gr.Button("Start transcription", variant="primary")
                    run_youtube = gr.Button("Transcribe YouTube", variant="primary")
                    cancel = gr.Button("Cancel", variant="secondary")
                    unload = gr.Button("Unload model", variant="secondary")
                unload_status = gr.Markdown("")
                unload.click(_unload, outputs=unload_status, queue=False)
            with gr.Column(scale=2):
                status = gr.Markdown("Upload files and start a transcription.")
                transcript = gr.Markdown()
                with gr.Row():
                    bundle = gr.File(label="All outputs (ZIP)")
                    json_file = gr.File(label="First file JSON")
                with gr.Row():
                    srt_file = gr.File(label="First file SRT")
                    vtt_file = gr.File(label="First file VTT")
        option_inputs = [files, model, language, batch_size, diarize, summarize, redact_pii, clean_format]
        file_event = run.click(
            _run_files,
            inputs=option_inputs,
            outputs=[status, transcript, bundle, json_file, srt_file, vtt_file],
        )
        youtube_event = run_youtube.click(
            _run_youtube,
            inputs=[youtube_url, model, language, batch_size, diarize, summarize, redact_pii, clean_format],
            outputs=[status, transcript, bundle, json_file, srt_file, vtt_file],
        )
        cancel.click(_request_cancel, outputs=unload_status, cancels=[file_event, youtube_event], queue=False)
        gr.Markdown(
            "Parakeet supplies timestamped subtitles. Nemotron is offered for broader language coverage but deliberately does not "
            "produce fabricated SRT or VTT timing. Optional extras (diarization, summary, PII redaction) run locally after ASR."
        )
    return app.queue(default_concurrency_limit=1)
