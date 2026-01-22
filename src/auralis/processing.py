import numpy as np
import torch
import torchaudio
from .config import SAMPLE_RATE, DEVICE, N_MELS, TARGET_LEN
from pydub import AudioSegment
import torch.nn.functional as F

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate = SAMPLE_RATE,
    n_mels = N_MELS,
    n_fft = 400,
    hop_length = 256,
    
).to(DEVICE)

amp_to_db = torchaudio.transforms.AmplitudeToDB().to(DEVICE)

class AudioLoadError(Exception):
    pass

def load_audio(path: str) -> torch.Tensor:
    waveform = None
    sr = None

    try:
        waveform, sr = torchaudio.load(path)
    except Execption as e1:
        try:
            audio = AudioSegment.from_file(path)
            audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

            if samples.size == 0:
                raise AudioLoadError("Empty audio file")

            waveform = torch.from_numpy(samples)
            sr = SAMPLE_RATE
        except Exception as e2:
            raise AudioLoadError(f"Failed to decode audio file: {str(e2)}") from e2

    if waveform is None or waveform.numel() == 0:
        raise AudioLoadError("Failed to load audio or audio is empty")

    if waveform.dim() > 1:
        waveform = waveform.mean(dim = 0)

    if sr!=SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    if waveform.numel() < TARGET_LEN:
        raise AudioLoadError("Audio too short for analysis")

    if waveform.numel() > TARGET_LEN:
        waveform = waveform[:TARGET_LEN]
    else:
        waveform = F.pad(waveform, (0, TARGET_LEN - waveform.numel()))

    return waveform.float()

def waveform_to_mel(waveform: torch.Tensor):
    mel = mel_transform(waveform.unsqueeze(0))
    mel = amp_to_db(mel)
    mel = mel.transpose(1, 2)
    return mel

def pad_time_dim(mel):
    T = mel.shape[1]
    pad_len = (8 - (T % 8)) % 8
    if pad_len > 0:
        mel = F.pad(mel, (0,0,0,pad_len))
    return mel

def extract_features(wav: torch.Tensor) -> torch.Tensor:
    mel = mel_transform(wav.unsqueeze(0))
    mel = amp_to_db(mel)
    if mel.dim == 4:
        mel = mel.squeeze(1)

    mel.transpose(1, 2)  # [B, T, N_MELS]
    mel = pad_time_dim(mel)
    return mel