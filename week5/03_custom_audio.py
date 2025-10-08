import torch
import torchaudio

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model pipeline
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
labels = bundle.get_labels()

# Decoder
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission):
        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

decoder = GreedyCTCDecoder(labels)

# Load your audio
path = "../audio_dataset/custom_audio_week5.wav"  # Adjust path as needed
waveform, sr = torchaudio.load(path)
if sr != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

waveform = waveform.to(device)

# Run model
with torch.inference_mode():
    emission, _ = model(waveform)
    transcript = decoder(emission[0])

print("TRANSCRIPT:", transcript)
