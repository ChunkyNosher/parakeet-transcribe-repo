from __future__ import annotations

import traceback
from html import escape
from pathlib import Path
from threading import Event
from urllib.parse import quote

import gradio as gr

from .backend import parse_key_phrases
from .diagnostics import doctor_report
from .exports import create_run_directory, readable_summary, write_bundle, write_result
from .models import DEFAULT_MODEL_KEY, MODELS, get_model
from .service import MAX_BATCH_SIZE, TranscriptionService
from .types import CancelledError, TranscriptionError, TranscriptResult

SERVICE = TranscriptionService()
CANCEL_REQUESTED = Event()

_EMPTY_PREVIEW_AND_FILES = (
    "<p></p>",
    [],
    None,
    None,
    None,
    None,
)


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
        f"Backend: NVIDIA NeMo (Docker Linux GPU)  \n"
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


def _gradio_file_url(path: Path) -> str:
    return f"/gradio_api/file={quote(str(path.resolve()))}"


def _caption_preview_html(audio_path: Path | None, vtt_path: Path | None) -> str:
    if audio_path is None or not audio_path.is_file():
        return "<p><em>No timed caption preview available for this run.</em></p>"
    audio_src = escape(_gradio_file_url(audio_path))
    if vtt_path is None or not vtt_path.is_file():
        return (
            f'<audio controls style="width:100%" src="{audio_src}"></audio>'
            "<p><em>Audio is ready, but there are no VTT cues "
            "(NeMo did not return native segment timestamps).</em></p>"
        )
    vtt_src = escape(_gradio_file_url(vtt_path))
    return (
        '<audio controls style="width:100%">'
        f'<source src="{audio_src}" type="audio/wav" />'
        f'<track kind="captions" src="{vtt_src}" srclang="en" label="Captions" default />'
        "</audio>"
        "<p><em>Captions use NeMo native segment timestamps. "
        "Toggle captions on the player if they are hidden.</em></p>"
    )


def _cue_table_rows(result: TranscriptResult) -> list[list[object]]:
    return [
        [round(segment.start, 3), round(segment.end, 3), segment.text]
        for segment in result.segments
    ]


def _failure_outputs(message: str) -> tuple:
    return (message, "", *_EMPTY_PREVIEW_AND_FILES)


def _publish_results(
    results: list[TranscriptResult], run_dir: Path
) -> tuple[str, str, str, list[list[object]], str | None, str | None, str | None, str | None]:
    artifacts = [write_result(result, run_dir) for result in results]
    bundle = write_bundle(results, run_dir)
    transcript = "\n\n".join(f"## {result.source_name}\n\n{result.text}" for result in results)
    status_parts = [f"- {readable_summary(result)}" for result in results]
    for result in results:
        if result.summary:
            status_parts.append(f"- Summary ready for {result.source_name}")
        if any(segment.speaker for segment in result.segments):
            status_parts.append(f"- Speaker labels attached for {result.source_name}")
        if result.runtime.get("key_phrase_count"):
            status_parts.append(f"- Keyterm boosting applied ({result.runtime['key_phrase_count']} phrases)")
        if result.runtime.get("segment_source") == "nemo_native":
            status_parts.append(f"- Caption cues from NeMo native segments ({len(result.segments)})")
        for warning in result.warnings:
            status_parts.append(f"- Warning ({result.source_name}): {warning}")
    status = "\n".join(status_parts)
    first_result = results[0]
    first = artifacts[0]
    preview_audio = first_result.runtime.get("preview_audio_path")
    audio_path = Path(preview_audio) if preview_audio else None
    vtt_path = Path(first["vtt"]) if first.get("vtt") else None
    preview_html = _caption_preview_html(audio_path, vtt_path)
    cue_rows = _cue_table_rows(first_result)
    return (
        status,
        transcript,
        preview_html,
        cue_rows,
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
    keyterms: str,
    boost_alpha: float,
    diarize: bool,
    summarize: bool,
    redact_pii: bool,
    clean_format: bool,
    progress: gr.Progress = gr.Progress(),  # noqa: B008 - Gradio injects this callback.
) -> tuple:
    try:
        CANCEL_REQUESTED.clear()
        run_dir = create_run_directory()
        results = SERVICE.transcribe_files(
            files or [],
            model_key=model_key,
            language=language.strip() or "auto",
            batch_size=int(batch_size),
            key_phrases=parse_key_phrases(keyterms),
            boost_alpha=float(boost_alpha),
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
        return _failure_outputs(str(exc))
    except TranscriptionError as exc:
        return _failure_outputs(f"### Transcription failed\n\n{exc}")
    except Exception as exc:  # pragma: no cover - defensive UI boundary
        traceback.print_exc()
        return _failure_outputs(f"### Unexpected failure\n\n`{type(exc).__name__}: {exc}`")


def _run_youtube(
    url: str,
    model_key: str,
    language: str,
    batch_size: int,
    keyterms: str,
    boost_alpha: float,
    diarize: bool,
    summarize: bool,
    redact_pii: bool,
    clean_format: bool,
    progress: gr.Progress = gr.Progress(),  # noqa: B008 - Gradio injects this callback.
) -> tuple:
    try:
        CANCEL_REQUESTED.clear()
        run_dir = create_run_directory()
        results = SERVICE.transcribe_youtube(
            url,
            model_key=model_key,
            language=language.strip() or "auto",
            batch_size=int(batch_size),
            key_phrases=parse_key_phrases(keyterms),
            boost_alpha=float(boost_alpha),
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
        return _failure_outputs(str(exc))
    except TranscriptionError as exc:
        return _failure_outputs(f"### Transcription failed\n\n{exc}")
    except Exception as exc:  # pragma: no cover - defensive UI boundary
        traceback.print_exc()
        return _failure_outputs(f"### Unexpected failure\n\n`{type(exc).__name__}: {exc}`")


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Parakeet Transcribe") as app:
        gr.Markdown(
            "# Parakeet Transcribe\n"
            "Local file transcription with **NVIDIA NeMo** (Parakeet / Nemotron). "
            "Supported runtime: Docker Compose Linux GPU container. "
            "Weights download once into the host-mounted cache; the default model is warmed into VRAM at startup "
            "(use **Unload model** to free VRAM)."
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
                    interactive=False,
                    info="Automatic language detection only for now. Locale forcing is not wired into NeMo yet.",
                )
                keyterms = gr.Textbox(
                    label="Keyterms (GPU-PB phrase boosting)",
                    lines=3,
                    placeholder="ProperNoun\nAcronym\nmulti word phrase",
                    info="One phrase per line (or comma-separated). Applied via NeMo GPU-PB shallow fusion. "
                    "Phrases are Title-Cased for Parakeet; leave empty to disable.",
                )
                boost_alpha = gr.Slider(
                    0.0,
                    5.0,
                    value=1.0,
                    step=0.1,
                    label="Keyterm boost strength",
                    info="NeMo boosting_tree_alpha. Higher biases harder toward listed phrases.",
                )
                batch_size = gr.Slider(
                    1,
                    MAX_BATCH_SIZE,
                    value=2,
                    step=1,
                    label="Chunk batch size (OOM fallback)",
                    info=f"Used when long-form local attention OOMs and the app falls back to chunking "
                    f"(max {MAX_BATCH_SIZE}). Leave the model loaded between files for best throughput.",
                )
                diarize = gr.Checkbox(
                    label="Speaker diarization (Sortformer GPU, MFCC fallback)",
                    value=False,
                    info="Unloads ASR briefly, runs NeMo Sortformer on CUDA, then falls back to CPU MFCC if needed. Best with Parakeet timestamps.",
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
                gr.Markdown("### Timed caption preview")
                caption_preview = gr.HTML(
                    value="<p><em>Timed caption preview appears here after a Parakeet run with native segments.</em></p>",
                )
                cue_table = gr.Dataframe(
                    headers=["Start (s)", "End (s)", "Text"],
                    datatype=["number", "number", "str"],
                    label="Caption cues (NeMo native segments)",
                    interactive=False,
                    wrap=True,
                )
                transcript = gr.Markdown(label="Full transcript")
                with gr.Row():
                    bundle = gr.File(label="All outputs (ZIP)")
                    json_file = gr.File(label="First file JSON")
                with gr.Row():
                    srt_file = gr.File(label="First file SRT")
                    vtt_file = gr.File(label="First file VTT")
        option_inputs = [
            files,
            model,
            language,
            batch_size,
            keyterms,
            boost_alpha,
            diarize,
            summarize,
            redact_pii,
            clean_format,
        ]
        result_outputs = [
            status,
            transcript,
            caption_preview,
            cue_table,
            bundle,
            json_file,
            srt_file,
            vtt_file,
        ]
        file_event = run.click(
            _run_files,
            inputs=option_inputs,
            outputs=result_outputs,
        )
        youtube_event = run_youtube.click(
            _run_youtube,
            inputs=[
                youtube_url,
                model,
                language,
                batch_size,
                keyterms,
                boost_alpha,
                diarize,
                summarize,
                redact_pii,
                clean_format,
            ],
            outputs=result_outputs,
        )
        cancel.click(_request_cancel, outputs=unload_status, cancels=[file_event, youtube_event], queue=False)
        gr.Markdown(
            "NeMo enables long-form local attention, greedy CUDA-graph decoding, and optional GPU-PB keyterms. "
            "Parakeet supplies timestamped subtitles from NeMo native segments. Nemotron is offered for broader "
            "language coverage but does not fabricate SRT/VTT timing. Live microphone streaming, Riva/NIM, and "
            "cloud ASR wrappers remain out of scope. Optional extras (diarization, summary, PII redaction) run "
            "locally after ASR."
        )
    return app.queue(default_concurrency_limit=1)
