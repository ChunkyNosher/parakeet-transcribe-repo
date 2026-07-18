from parakeet_transcribe.models import DEFAULT_MODEL_KEY, MODELS, get_model


def test_model_registry_has_capability_boundaries() -> None:
    assert get_model(DEFAULT_MODEL_KEY).capabilities.timestamps
    assert not MODELS["nemotron-3.5"].capabilities.timestamps


def test_english_tdt_variants_are_timestamped_single_language() -> None:
    for key in ("parakeet-v2", "parakeet-1.1b"):
        spec = get_model(key)
        assert spec.model_class == "tdt"
        assert spec.capabilities.timestamps
        assert not spec.capabilities.automatic_language_detection
        assert spec.capabilities.supported_languages == 1


def test_model_registry_keys_and_ids_are_unique() -> None:
    assert len(MODELS) == len(set(MODELS))
    assert len({spec.model_id for spec in MODELS.values()}) == len(MODELS)


def test_only_parakeet_1_1b_uses_lowercase_vocab() -> None:
    assert get_model("parakeet-1.1b").capabilities.lowercase_vocab is True
    for key in ("parakeet-v3", "parakeet-v2", "nemotron-3.5"):
        assert get_model(key).capabilities.lowercase_vocab is False
