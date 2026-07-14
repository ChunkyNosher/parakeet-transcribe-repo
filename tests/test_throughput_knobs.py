from parakeet_transcribe.service import MAX_BATCH_SIZE


def test_max_batch_size_raised() -> None:
    assert MAX_BATCH_SIZE >= 8
