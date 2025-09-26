# week4/4_bonus_features.py
import os, torch, torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from utils_audio import DATA_DIR, load_audio_ffmpeg, iter_dataset_m4a, plot_matrix, DEFAULT_SR

SR = DEFAULT_SR
N_FFT = 400
WIN = 400
HOP = 160
N_MELS = 80

def main():
    for fname, path in iter_dataset_m4a():
        wf, sr = load_audio_ffmpeg(path, SR, mono=True)

        # 1) Mel Filter Bank 
        n_freqs = N_FFT // 2 + 1
        mel_fb = F.melscale_fbanks(
            n_freqs, n_mels=N_MELS, f_min=0.0, f_max=sr/2.0,
            sample_rate=sr, norm="slaney", mel_scale="htk",
        )
        plot_matrix(mel_fb.T, f"Mel Filter Bank - {fname}", ylabel="mel bins")

        # 2) Mel-Spectrogram
        mel = T.MelSpectrogram(
            sample_rate=sr, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
            n_mels=N_MELS, f_min=0.0, f_max=sr/2.0, norm="slaney", mel_scale="htk", power=2.0
        )(wf)
        plot_matrix(mel[0].log2().detach().numpy(), f"MelSpectrogram - {fname}", ylabel="mel freq")

        # 3) MFCC
        mfcc = T.MFCC(
            sample_rate=sr, n_mfcc=13,
            melkwargs={"n_fft": N_FFT, "win_length": WIN, "hop_length": HOP,
                       "n_mels": N_MELS, "f_min": 0.0, "f_max": sr/2.0,
                       "norm": "slaney", "mel_scale": "htk", "power": 2.0}
        )(wf)
        plot_matrix(mfcc[0].detach().numpy(), f"MFCC - {fname}", ylabel="coeff idx")

        # 4) LFCC
        lfcc = T.LFCC(
            sample_rate=sr, n_lfcc=20,
            speckwargs={"n_fft": N_FFT, "win_length": WIN, "hop_length": HOP, "power": 2.0}
        )(wf)
        plot_matrix(lfcc[0].detach().numpy(), f"LFCC - {fname}", ylabel="coeff idx")

        # 5) Pitch
        pitch = F.detect_pitch_frequency(wf, sr)  # (1,T)

if __name__ == "__main__":
    main()
