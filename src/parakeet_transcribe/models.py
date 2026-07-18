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

PARAKEET_V2 = ModelSpec(
    key="parakeet-v2",
    label="Parakeet TDT 0.6B v2 — English only, timestamps, best English WER",
    model_id="nvidia/parakeet-tdt-0.6b-v2",
    model_class="tdt",
    capabilities=ModelCapabilities(
        timestamps=True, automatic_language_detection=False, supported_languages=1
    ),
)

PARAKEET_11B = ModelSpec(
    key="parakeet-1.1b",
    label="Parakeet TDT 1.1B — English only, timestamps, punctuation auto-restored",
    model_id="nvidia/parakeet-tdt-1.1b",
    model_class="tdt",
    capabilities=ModelCapabilities(
        timestamps=True,
        automatic_language_detection=False,
        supported_languages=1,
        lowercase_vocab=True,
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

MODELS: dict[str, ModelSpec] = {
    spec.key: spec for spec in (PARAKEET_V3, PARAKEET_V2, PARAKEET_11B, NEMOTRON_35)
}
DEFAULT_MODEL_KEY = PARAKEET_V3.key


def get_model(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown model key: {key}") from exc
