import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
import os
import librosa
import math

from nv_lib.model import MelSpectrogramConfig


class WavMelDataset(Dataset):
    def __init__(self, mel_dir, wav_dir=None, limit=None):
        super().__init__()
        self.mel_dir = mel_dir
        self.wav_dir = wav_dir
        self.names = [path[:-4] for path in os.listdir(mel_dir) if path[-4:] == '.npy']

    def __getitem__(self, index):
        name = self.names[index]

        mel = np.load(os.path.join(self.mel_dir, f"{name}.npy"))
        mel = torch.from_numpy(mel)
        elem = {
            "mel": mel,
        }

        if self.wav_dir is not None:
            wav, _ = librosa.load(os.path.join(self.wav_dir, f"{name}.wav"))
            wav = torch.from_numpy(wav)
            elem.update({
                "wav": wav,
            })

        return elem
    
    def __len__(self):
        return len(self.names)
