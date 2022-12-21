import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
import os
import librosa
import random


class WavDataset(Dataset):
    def __init__(self, wav_dir, max_wav_len=None, limit=None, preload=False):
        super().__init__()
        self.wav_dir = wav_dir
        self.names = [path[:-4] for path in os.listdir(wav_dir) if path[-4:] == '.wav']

        if preload:
            self.preloaded_wavs = []
            for name in self.names:
                wav, _ = librosa.load(os.path.join(self.wav_dir, f"{name}.wav"))
                wav = torch.from_numpy(wav)
                self.preloaded_wavs.append(wav)
        else:
            self.preloaded_wavs = None

        self.max_wav_len = max_wav_len

    def __getitem__(self, index):
        if self.preloaded_wavs is None:
            name = self.names[index]
            wav, _ = librosa.load(os.path.join(self.wav_dir, f"{name}.wav"))
            wav = torch.from_numpy(wav)
        else:
            wav = self.preloaded_wavs[index]

        if self.max_wav_len is not None:
            if wav.shape[0] >= self.max_wav_len:
                max_wav_start = wav.shape[0] - self.max_wav_len
                wav_start = random.randint(0, max_wav_start)
                wav = wav[wav_start:wav_start+self.max_wav_len]
            else:
                wav = torch.nn.functional.pad(wav, (0, self.max_wav_len - wav.shape[0]), 'constant')

        # wav = wav * 0.95
        return {
            "gt_wav": wav,
        }
    
    def __len__(self):
        return len(self.names)
