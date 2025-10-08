import torch
import torchaudio
import matplotlib.pyplot as plt
import IPython.display as ipd

# Backend for mp3/wav if needed
torchaudio.set_audio_backend("soundfile")

# Show versions
print("Torch:", torch.__version__)
print("Torchaudio:", torchaudio.__version__)

# Set seed and device
torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 1. Download official tutorial sample audio
from torchaudio.utils import download_asset
SPEECH_FILE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")

# 2. Create model pipeline
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
print("Sample Rate:", bundle.sample_rate)
print("Labels:", bundle.get_labels())
model = bundle.get_model().to(device)

# 3. Load audio
waveform, sample_rate = torchaudio.load(SPEECH_FILE)
waveform = waveform.to(device)
if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

# 4. Play audio
print("▶️ Playing audio...")
ipd.display(ipd.Audio(SPEECH_FILE))

# 5. Extract acoustic features from all 12 layers
with torch.inference_mode():
    features, _ = model.extract_features(waveform)

print(f"Number of transformer layers: {len(features)}")

# 6. Plot feature maps of all 12 layers
fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
for i, feats in enumerate(features):
    ax[i].imshow(feats[0].cpu(), interpolation="nearest", aspect='auto', origin='lower')
    ax[i].set_title(f"Feature from transformer layer {i+1}")
    ax[i].set_xlabel("Feature dimension")
    ax[i].set_ylabel("Frame (time-axis)")
fig.tight_layout()
plt.savefig("all_layers_features.png")
plt.show()
print("✅ Saved: all_layers_features.png")

# 7. Feature classification (logits)
with torch.inference_mode():
    emission, _ = model(waveform)

# 8. Plot logits
plt.figure(figsize=(10, 4))
plt.imshow(emission[0].cpu().T, interpolation="nearest", aspect='auto', origin='lower')
plt.title("Classification result (logits)")
plt.xlabel("Time step")
plt.ylabel("Class label")
plt.tight_layout()
plt.savefig("classification_logits.png")
plt.show()

# 9. Define greedy CTC decoder
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

# 10. Decode and show result
decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])
print("📜 Transcript:", transcript)
