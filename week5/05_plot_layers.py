import torch
import torchaudio
import matplotlib.pyplot as plt
import os

# Backend for mp3/wav
torchaudio.set_audio_backend("soundfile")

# Load model
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()
model.eval()

# Load your custom audio file
path = "../audio_dataset/custom_audio_week5.wav"
assert os.path.exists(path), "Audio file not found!"
waveform, sr = torchaudio.load(path)

# Resample if needed
if sr != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

# Extract features from all transformer layers
with torch.inference_mode():
    features, _ = model.extract_features(waveform)

# Plot ALL layers
for i, feat in enumerate(features):
    data = feat[0].cpu().T  # Shape: [dim, time]
    plt.figure(figsize=(12, 4))
    plt.imshow(data, aspect="auto", origin="lower")
    plt.title(f"Feature Map - Transformer Layer {i+1}")
    plt.xlabel("Time Steps")
    plt.ylabel("Feature Dimension")
    plt.colorbar()
    plt.tight_layout()

    fname = f"custom_feat_layer_{i+1:02}.png"
    plt.savefig(fname)
    print(f"✅ Saved: {fname}")
    plt.close()
