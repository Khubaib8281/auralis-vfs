import numpy as np
from auralis.scorer import score_waveform

def test_score_waveform_range():
    fake_audio = np.random.randn(16000).astype("float32")
    score = score_waveform(fake_audio)

    assert isinstance(score, float)

    assert 0.0 <= score <= 100.0