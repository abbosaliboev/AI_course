import torchaudio
import matplotlib.pyplot as plt
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import os

# Set audio backend if needed (optional)
torchaudio.set_audio_backend("soundfile")

# Load your audio (same as in 03_custom_audio.py)
path = "../audio_dataset/custom_audio_week5.wav"
if not os.path.exists(path):
    raise FileNotFoundError(f"Audio file not found: {path}")

waveform, sr = torchaudio.load(path)

# Resample to 16kHz if needed
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

# Apply MelSpectrogram
mel = MelSpectrogram(sample_rate=16000, n_mels=64)(waveform)
db = AmplitudeToDB()(mel)

# Plot
plt.figure(figsize=(10, 4))
plt.imshow(db[0].cpu(), origin='lower', aspect='auto')
plt.title("Mel Spectrogram (log scale)")
plt.colorbar(format='%+2.0f dB')
plt.xlabel("Time")
plt.ylabel("Mel bins")
plt.tight_layout()
plt.show()
