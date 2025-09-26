# week4/2_resampling.py
import os
import torchaudio
import utils_audio
from utils_audio import DATA_DIR, load_audio_ffmpeg, iter_dataset_m4a, plot_waveform, plot_matrix


def main():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"DATA_DIR is not find: {DATA_DIR}")

    for fname, path in iter_dataset_m4a():

        wf_16k, sr16 = load_audio_ffmpeg(path, 16000, mono=True)

        # Downsample: 16k -> 8k
        down = torchaudio.transforms.Resample(orig_freq=sr16, new_freq=8000)(wf_16k)

        # Upsample: 8k -> 16k
        up = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)(down)

        print(f">>> {fname}")
        print(f"   16k shape: {tuple(wf_16k.shape)}")
        print(f"   8k  shape: {tuple(down.shape)}")
        print(f"   up  shape: {tuple(up.shape)}")

if __name__ == "__main__":
    main()
