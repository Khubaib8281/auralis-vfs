import numpy as np
from auralis.scorer import score_waveform, score_audio

def test_score_waveform_range():
    fake_audio = np.random.randn(16000).astype("float32")
    score_wav = score_waveform(fake_audio)
    score_voice = score_audio("/mnt/d/CODE/AI/vocal_fatigue_detection_research/voices/sample.wav")

    assert isinstance(score_wav, float)
    assert isinstance(score_voice, float)

    assert 0.0 <= score_wav <= 100.0
    assert 0.0 <= score_voice <= 100.0