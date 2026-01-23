import numpy as np
from auralis.scorer import score_waveform, score_audio
import soundfile as sf
import warnings
from unittest.mock import patch

def pytest_configure():
    warnings.filterwarnings(
        "ignore", 
        message = "builtin type SwigPy.* has no __module__ attribute",
        category = DeprecationWarning,
    )

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


def test_score_audio_mocked():
    with patch("auralis.processing.load_audio") as mock_load:
        mock_load.return_value = (torch.randn(1, 16000), 16000)

        score = score_audio("dummy.wav")
        assert 0.0 <= score <= 100.0