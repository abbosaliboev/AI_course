# week4/1_waveform_spectrogram.py
import os
import torchaudio
from utils_audio import (
    DATA_DIR,
    load_audio_ffmpeg,
    iter_dataset_m4a,
    plot_waveform,
    plot_matrix,
)


N_FFT = 400
WIN = 400
HOP = 160

def main():
    print("SCRIPT=1_waveform_spectrogram")  
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"DATA_DIR topilmadi: {DATA_DIR}")

    spec_tf = torchaudio.transforms.Spectrogram(
        n_fft=N_FFT, win_length=WIN, hop_length=HOP, power=2.0
    )

    for fname, path in iter_dataset_m4a():
        wf, sr = load_audio_ffmpeg(path, 16000, mono=True)
        print(f">>> {fname}  wf={tuple(wf.shape)}  sr={sr}")

        # 1) Waveform
        plot_waveform(wf, sr, title=f"Waveform - {fname}")

        # 2) Spectrogram
        spec = spec_tf(wf)  # (1, n_freq, T)
        plot_matrix(spec[0].log2().numpy(), f"Spectrogram - {fname}", ylabel="freq bin")

if __name__ == "__main__":
    main()
