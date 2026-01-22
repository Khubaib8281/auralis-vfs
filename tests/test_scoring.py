import numpy as np
from auralis.scorer import score_waveform, score_audio
import soundfile as sf

def test_score_waveform():
    fake_audio = np.random.randn(16000).astype("float32")
    score_wav = score_waveform(fake_audio)

    assert isinstance(score_wav, float)

    assert 0.0 <= score_wav <= 100.0

def test_audio_score(tmp_path):
    sr = 16000
    duration_sec = 5

    audio = np.random.randn(sr * duration_sec).astype("float32")

    wav_path = tmp_path / "sample.wav"
    sf.write(wav_path, audio, sr)

    score = score_audio(str(wav_path))

    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0