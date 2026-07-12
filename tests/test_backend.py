from parakeet_transcribe.backend import _extract_language, _words_from_timestamp_payload


def test_tdt_token_spans_align_to_visible_words() -> None:
    payload = [
        {"token": "W", "start": 0.32, "end": 0.40},
        {"token": "ell", "start": 0.40, "end": 0.56},
        {"token": ",", "start": 0.56, "end": 0.56},
        {"token": "I", "start": 0.64, "end": 0.80},
        {"token": "don", "start": 0.80, "end": 0.96},
        {"token": "'t", "start": 0.96, "end": 1.04},
    ]
    words = _words_from_timestamp_payload(payload, "Well, I don't")
    assert [(word.text, word.start, word.end) for word in words] == [
        ("Well,", 0.32, 0.56),
        ("I", 0.64, 0.80),
        ("don't", 0.80, 1.04),
    ]


def test_language_tag_is_removed_from_nemotron_transcript() -> None:
    assert _extract_language("Bonjour tout le monde. <fr-FR>") == ("Bonjour tout le monde.", "fr-FR")
