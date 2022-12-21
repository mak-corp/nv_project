import argparse
import librosa
import numpy as np
import os
import pathlib
from tqdm import tqdm
import torch

from nv_lib.model import MelSpectrogram


SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()


def generate_mels(wav_dir, mel_dir, limit=None):
    print()
    print("============================== Processing started ==============================")
    print()

    mel_generator = MelSpectrogram()

    os.makedirs(mel_dir, exist_ok=False)

    for i, file in enumerate(tqdm(os.listdir(wav_dir))):
        if file[-4:] != '.wav':
            continue
        wav, rate = librosa.load(os.path.join(wav_dir, file))
        assert rate == mel_generator.config.sr

        wav = torch.from_numpy(wav).unsqueeze(0)

        mel = mel_generator(wav)
        mel = mel.squeeze(0).numpy()

        np.save(os.path.join(mel_dir, file.replace('.wav', '.npy')), mel, allow_pickle=False)
        if limit is not None and i + 1 >= limit:
            break

    print()
    print("============================== Processing finished ==============================")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-dir", type=str, required=True)
    parser.add_argument("--mel-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, required=False)
    args = parser.parse_args()

    generate_mels(args.wav_dir, args.mel_dir, args.limit)
