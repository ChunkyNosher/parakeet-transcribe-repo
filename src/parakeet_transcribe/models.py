from .types import ModelCapabilities, ModelSpec

PARAKEET_V3 = ModelSpec(
    key="parakeet-v3",
    label="Parakeet TDT 0.6B v3 — timestamps, 25 languages",
    model_id="nvidia/parakeet-tdt-0.6b-v3",
    model_class="tdt",
    capabilities=ModelCapabilities(
        timestamps=True, automatic_language_detection=True, supported_languages=25
    ),
)

NEMOTRON_35 = ModelSpec(
    key="nemotron-3.5",
    label="Nemotron 3.5 ASR 0.6B — 32 locales, no timestamps",
    model_id="nvidia/nemotron-3.5-asr-streaming-0.6b",
    model_class="rnnt",
    capabilities=ModelCapabilities(
        timestamps=False, automatic_language_detection=True, supported_languages=32
    ),
)

MODELS: dict[str, ModelSpec] = {spec.key: spec for spec in (PARAKEET_V3, NEMOTRON_35)}
DEFAULT_MODEL_KEY = PARAKEET_V3.key


def get_model(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown model key: {key}") from exc
