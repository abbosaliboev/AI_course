# week4/utils_audio.py
import os, subprocess, tempfile
import torch, torchaudio
import matplotlib.pyplot as plt

DATA_DIR = r"../audio_dataset"
DEFAULT_SR = 16000

def load_audio_ffmpeg(path: str, target_sr: int = DEFAULT_SR, mono: bool = True):
    """m4a/mp3 kabi formatlarni FFmpeg orqali vaqtincha WAV ga konvert qilib, torchaudio bilan o‘qiydi."""
    with tempfile.TemporaryDirectory() as td:
        tmp_wav = os.path.join(td, "tmp.wav")
        cmd = ["ffmpeg", "-y", "-i", path]
        if mono:
            cmd += ["-ac", "1"]
        cmd += ["-ar", str(target_sr), tmp_wav]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        wf, sr = torchaudio.load(tmp_wav)  # (C,N)
        return wf, sr  # sr == target_sr

def iter_dataset_m4a(data_dir: str = DATA_DIR):
    for name in sorted(os.listdir(data_dir)):
        if name.lower().endswith(".m4a"):
            yield name, os.path.join(data_dir, name)

def plot_waveform(waveform: torch.Tensor, sr: int, title: str, show: bool = True, save_path: str | None = None):
    plt.figure(figsize=(10, 3))
    plt.plot(waveform[0].numpy())
    plt.title(f"{title} (SR={sr})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    elif show:
        plt.show()

def plot_matrix(M, title: str, ylabel: str, show: bool = True, save_path: str | None = None):
    plt.figure(figsize=(10, 4))
    plt.imshow(M, aspect="auto", origin="lower")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    elif show:
        plt.show()
