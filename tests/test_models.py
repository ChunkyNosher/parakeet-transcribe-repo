from parakeet_transcribe.models import DEFAULT_MODEL_KEY, MODELS, get_model


def test_model_registry_has_capability_boundaries() -> None:
    assert get_model(DEFAULT_MODEL_KEY).capabilities.timestamps
    assert not MODELS["nemotron-3.5"].capabilities.timestamps
