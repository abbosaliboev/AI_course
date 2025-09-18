import torchaudio
import matplotlib.pyplot as plt
from torchaudio.utils import download_asset
from torchaudio.transforms import Spectrogram

# 1) Namuna audio faylni yuklab olish (torchaudio tutorial asset)
SAMPLE_WAV = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")

# 2) Audio o‘qish (waveform, sample_rate)
waveform, sample_rate = torchaudio.load(SAMPLE_WAV)  # shape: [channels, time]
print("waveform shape:", waveform.shape, "sample_rate:", sample_rate)

# 3) Waveform chizish
plt.figure(figsize=(10, 3))
plt.title("Waveform")
plt.plot(waveform.t().numpy())
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("waveform.png", dpi=150)
# plt.show()  # GUI bo'lmasa, saqlash kifoya

# 4) Spectrogram hisoblash va chizish
spec = Spectrogram(n_fft=400, win_length=400, hop_length=160, power=2.0)
S = spec(waveform)  # shape: [channels, freq, time]

# Mono bo'lsa 0-kanalni olamiz
S_db = 10 * torchaudio.functional.amplitude_to_DB(S, multiplier=10.0, amin=1e-10, db_multiplier=0.0)

plt.figure(figsize=(10, 4))
plt.title("Spectrogram (dB)")
plt.imshow(S_db[0].numpy(), origin="lower", aspect="auto")
plt.xlabel("Frames")
plt.ylabel("Frequency bins")
plt.colorbar(label="dB")
plt.tight_layout()
plt.savefig("spectrogram.png", dpi=150)
# plt.show()

print("Saved: waveform.png, spectrogram.png")
