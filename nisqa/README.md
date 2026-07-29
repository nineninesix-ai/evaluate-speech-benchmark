# NISQA: Speech Quality and Naturalness Assessment (Inference Only)

**Inference-only module for speech quality prediction using deep learning.**

NISQA is a deep learning model for speech quality prediction that provides:
- **Overall Quality (MOS)**: Mean Opinion Score prediction
- **Quality Dimensions**: Noisiness, Coloration, Discontinuity, and Loudness
- **TTS Naturalness**: Prediction for synthesized speech

This is a refactored inference-only version - training and evaluation code has been removed for simplicity.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

Requirements:
- Python 3.7+
- PyTorch 1.10.0+
- librosa, numpy, scipy, soundfile

## Quick Start

```python
from nisqa import NisqaModel

# Initialize model
model = NisqaModel()

# Predict from file
result = model(filepath='path/to/audio.wav')

print(f"MOS: {result['mos_pred']:.2f}")
print(f"Noisiness: {result['noi_pred']:.2f}")
print(f"Coloration: {result['col_pred']:.2f}")
print(f"Discontinuity: {result['dis_pred']:.2f}")
print(f"Loudness: {result['loud_pred']:.2f}")
```

## Usage

### Initialize Model

```python
from nisqa import NisqaModel

# Use default model (NISQA v2.0 with multidimensional predictions)
model = NisqaModel()

# Or specify model path
model = NisqaModel(model_path='weights/nisqa.tar')

# Use NISQA-TTS for synthesized speech
model_tts = NisqaModel(model_path='weights/nisqa_tts.tar')

# Specify device
model_gpu = NisqaModel(device='cuda')
model_cpu = NisqaModel(device='cpu')
```

### Predict Speech Quality

**From file path:**
```python
result = model(filepath='path/to/audio.wav')
```

**From numpy array:**
```python
import numpy as np

# Your audio waveform as numpy array (assumes 48kHz sample rate)
audio = np.random.randn(48000)  # Example: 1 second at 48kHz

result = model(waveform=audio)

# Or specify sample rate explicitly
import librosa
audio, sr = librosa.load('audio.wav', sr=None)
result = model(waveform=audio, sr=sr)
```

**From PyTorch tensor:**
```python
import torch

audio = torch.randn(48000)
result = model(waveform=audio, sr=48000)
```

### Output Format

The model returns a dictionary with predictions:

```python
{
    'mos_pred': 3.5,      # Overall quality (1-5)
    'noi_pred': 3.2,      # Noisiness (1-5)
    'dis_pred': 4.1,      # Discontinuity (1-5)
    'col_pred': 3.8,      # Coloration (1-5)
    'loud_pred': 3.9      # Loudness (1-5)
}
```

## Available Models

Three pretrained models are included in the `weights/` directory:

| Model | Use Case | Output | File |
|-------|----------|--------|------|
| NISQA v2.0 | Transmitted Speech | MOS + 4 dimensions | `weights/nisqa.tar` |
| NISQA v2.0 (MOS only) | Transmitted Speech | MOS only | `weights/nisqa_mos_only.tar` |
| NISQA-TTS | Synthesized Speech | Naturalness | `weights/nisqa_tts.tar` |

## Example

See `example.py` for complete usage examples.

## Model Architecture

NISQA uses a three-stage architecture:
1. **Framewise Model**: CNN or feed-forward network
2. **Time-Dependency Model**: Self-Attention or LSTM
3. **Pooling**: Attention, average, max, or last-step pooling

## Paper and Citation

If you use NISQA for your research, please cite:

```bibtex
@inproceedings{mittag21_interspeech,
  author={Gabriel Mittag and Babak Naderi and Assmaa Chehadi and Sebastian Möller},
  title={{NISQA: A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={2127--2131},
  doi={10.21437/Interspeech.2021-299}
}
```

## License

- Code: [MIT License](LICENSE)
- Model weights: [CC BY-NC-SA 4.0](weights/LICENSE_model_weights) (Non-commercial use)

## Original Repository

This is a refactored inference-only version. For the original repository with training capabilities, see:
https://github.com/gabrielmittag/NISQA

Copyright © 2021 Gabriel Mittag
www.qu.tu-berlin.de
