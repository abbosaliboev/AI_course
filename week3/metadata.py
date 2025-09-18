import torchaudio
from torchaudio.utils import download_asset

SAMPLE_WAV = download_asset(
    "tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
)

md = torchaudio.info(SAMPLE_WAV)
print(md)
