import numpy as np

from parakeet_transcribe.backend import max_new_tokens_for_audio
from parakeet_transcribe.service import MAX_BATCH_SIZE


def test_max_batch_size_raised() -> None:
    assert MAX_BATCH_SIZE >= 8


def test_max_new_tokens_scales_with_duration() -> None:
    short = [np.zeros(16_000, dtype=np.float32)]
    long = [np.zeros(16_000 * 60, dtype=np.float32)]
    assert max_new_tokens_for_audio(short) < max_new_tokens_for_audio(long)
    assert max_new_tokens_for_audio(long) <= 4096
