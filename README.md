# Auralis VFS (Vocal Fatigue Scoring Library)

[![PyPI](https://img.shields.io/pypi/v/auralis-vfs?style=flat-square)](https://pypi.org/project/auralis-vfs/)
[![Python](https://img.shields.io/pypi/pyversions/auralis-vfs?style=flat-square)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Overview

**Auralis VFS** is a research-grade Python library for **objective vocal fatigue assessment** using speech audio. It leverages **state-of-the-art deep learning models** (ECAPA-TDNN-based embeddings and supervised contrastive learning) to compute a **Vocal Fatigue Score (0–100)** from short audio recordings.

This library is designed for:

* Research studies in voice health, occupational voice monitoring, and speech pathology.
* Integration into speech analysis pipelines.
* Reproducible and standardized scoring across datasets.

**Cite our research:**

> Ahmad, M. K. (2026). Modeling Vocal Fatigue as Embedding-Space Deviation Using Contrastively Trained ECAPA-TDNNs (0.1.0). Zenodo. https://doi.org/10.5281/zenodo.18305757

---

## Key Features

* Compute **Vocal Fatigue Score** from raw audio (`.wav`, `.mp3`, `.m4a`).
* Fast waveform-based scoring using pretrained **ECAPA-TDNN embeddings**.
* Reference-based scoring using curated embeddings from healthy speakers.
* **Production-ready API** with `score_audio()` and `score_waveform()` functions.
* Configurable parameters for audio sampling rate, duration, and mel-spectrogram features.
* Designed for **research reproducibility**.

---

## Installation

```bash
pip install auralis-vfs
```

**Dependencies:**

* Python >= 3.10
* torch >= 2.1.1
* torchaudio >= 2.1.1
* speechbrain >= 1.0.3
* numpy >= 1.23
* soundfile
* scipy
* pydub
* PyYAML

> Optional: GPU acceleration works automatically if PyTorch detects a CUDA-enabled device.

---

## Usage

### 1. Scoring a waveform

```python
import numpy as np
from auralis.scorer import score_waveform

# Generate fake waveform (1 second of audio at 16kHz)
waveform = np.random.randn(16000).astype("float32")

score = score_waveform(waveform)
print(f"Vocal Fatigue Score: {score:.2f}")
```

### 2. Scoring an audio file

```python
from auralis.scorer import score_audio

audio_path = "path/to/speech_sample.wav"
score = score_audio(audio_path)
print(f"Vocal Fatigue Score: {score:.2f}")
```

>## Audio Validation

- Supported formats: .wav, .mp3, .m4a
- Duration: 5–10 seconds recommended

> Scores range from **0 (no fatigue)** to **100 (severe fatigue)**.

---

## File & Directory Structure

```
auralis-vfs/
├─ src/auralis/
│  ├─ __init__.py
│  ├─ scorer.py          # Public API functions
|  ├─ validators.py
│  ├─ ecapa.py           # Model wrapper
│  ├─ processing.py      # Audio & feature processing
│  ├─ config.py          # Paths & constants
│  ├─ data/              # Reference embeddings & axis
│  └─ models/            # Pretrained ECAPA-TDNN-VHE model & config.yaml
├─ tests/
│  ├─ test_scoring.py
├─ pyproject.toml
├─ setup.cfg
├─ CITATIONS.cff
├─ MANIFEST.in
├─ .gitignore
├─ README.md
├─ requirements.txt
└─ LICENSE
```

---

## API Reference

### `score_waveform(waveform: np.ndarray) -> float`

* `waveform`: 1D numpy array representing audio samples.
* Returns: Vocal Fatigue Score (float, 0–100).

### `score_audio(file_path: str) -> float`

* `file_path`: Path to audio file (`.wav`, `.mp3`, `.m4a`).
* Validates file extension and duration.
* Returns: Vocal Fatigue Score (float, 0–100).

---

## Future Work

Planned improvements to enhance auralis_vfs:

- **Prosody Feature Integration** – Analyze pitch, energy, and speaking rate to enrich scoring.

- **Clinical Report Generation** – Provide automatic reports resembling clinical assessments, including:

    - Fatigue trends over time

    - Prosody-based analysis

    - Summary interpretation for voice health monitoring


- **Web/API Interface** – Seamless integration with Gradio or FastAPI for cloud deployments.

## Contributors & Credits

**Authors / Maintainers:**

* **Muhammad Khubaib Ahmad** – AI/ML Architect, Vocal Fatigue Modeling

**Contributors:**

* **Faiez Ahmad(Data Manager)** – Dataset collection and preprocessing
* **Muhammad Anas Tariq(Data Collector)** – Dataset organization and verification

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LISENCE) file for details.

---

## Notes for Researchers

* Designed for **short audio clips** (5–10 seconds).
* Scores are **relative to healthy reference embeddings**.
* Reproducibility is guaranteed by **fixed model weights and configuration files**.
* Compatible with both **CPU and GPU** setups.

---

## Contact

* **Email**: [muhammadkhubaibahmad854@gmail.com](mailto:muhammadkhubaibahmad854@gmail.com)
* **GitHub**: [Khubaib8281/auralis-vfs](https://github.com/Khubaib8281/auralis-vfs)
