from __future__ import annotations

from threading import Event

import gradio as gr

from .diagnostics import doctor_report
from .exports import create_run_directory, readable_summary, write_bundle, write_result
from .models import DEFAULT_MODEL_KEY, MODELS, get_model
from .service import TranscriptionService
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


def _run(
    files: list[str] | None,
    model_key: str,
    language: str,
    batch_size: int,
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
        )
        artifacts = [write_result(result, run_dir) for result in results]
        bundle = write_bundle(results, run_dir)
        transcript = "\n\n".join(f"## {result.source_name}\n\n{result.text}" for result in results)
        status = "\n".join(f"- {readable_summary(result)}" for result in results)
        first = artifacts[0]
        return (
            status,
            transcript,
            str(bundle),
            str(first["json"]),
            str(first.get("srt")) if first.get("srt") else None,
            str(first.get("vtt")) if first.get("vtt") else None,
        )
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
                    label="Audio or video files",
                    file_count="multiple",
                    type="filepath",
                    file_types=["audio", "video"],
                )
                model = gr.Dropdown(_model_choices(), value=DEFAULT_MODEL_KEY, label="NVIDIA model")
                details = gr.Markdown(_model_details(DEFAULT_MODEL_KEY))
                model.change(_model_details, inputs=model, outputs=details)
                language = gr.Textbox(
                    value="auto",
                    label="Language",
                    info="Use auto, or a locale such as en-US or de-DE for Nemotron.",
                )
                batch_size = gr.Slider(1, 4, value=1, step=1, label="Chunk batch size")
                with gr.Row():
                    run = gr.Button("Start transcription", variant="primary")
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
        event = run.click(
            _run,
            inputs=[files, model, language, batch_size],
            outputs=[status, transcript, bundle, json_file, srt_file, vtt_file],
        )
        cancel.click(_request_cancel, outputs=unload_status, cancels=[event], queue=False)
        gr.Markdown(
            "Parakeet supplies timestamped subtitles. Nemotron is offered for broader language coverage but deliberately does not "
            "produce fabricated SRT or VTT timing."
        )
    return app.queue(default_concurrency_limit=1)
