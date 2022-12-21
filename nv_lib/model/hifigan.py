import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from dacite import from_dict
import numpy as np
import torch.nn.functional as F
import itertools

from .base_model import BaseModel
from .discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from .generator import Generator
from .wav_to_mel import MelSpectrogram

from nv_lib.utils import MetricTracker


@dataclass
class HiFiGANConfig:
    learning_rate: float = 2e-4
    adam_b1: float = 0.8
    adam_b2: float = 0.99
    lr_decay: float = 0.99
    

class HiFiGAN(BaseModel):
    """ FastSpeech """

    def __init__(self, **model_config):
        super(HiFiGAN, self).__init__()

        cfg = from_dict(data_class=HiFiGANConfig, data=model_config)
        self.model_config = cfg

        self.mel_generator = MelSpectrogram()

        self.generator = Generator()
        self.mp_discriminator = MultiPeriodDiscriminator()
        self.ms_discriminator = MultiScaleDiscriminator()

        self.optim_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=cfg.learning_rate,
            betas=[cfg.adam_b1, cfg.adam_b2]
        )
        self.optim_d = torch.optim.AdamW(
            itertools.chain(self.mp_discriminator.parameters(), self.ms_discriminator.parameters()),
            lr=cfg.learning_rate,
            betas=[cfg.adam_b1, cfg.adam_b2]
        )

        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=cfg.lr_decay)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=cfg.lr_decay)

    def extra_state_dict(self):
        return {
            "optim_g": self.optim_g.state_dict(),
            "optim_d": self.optim_d.state_dict(),
            "scheduler_g": self.scheduler_g.state_dict(),
            "scheduler_d": self.scheduler_d.state_dict(),
        }

    def load_extra_state_dict(self, state_dict):
        self.optim_g.load_state_dict(state_dict["optim_g"])
        self.optim_d.load_state_dict(state_dict["optim_d"])
        self.scheduler_g.load_state_dict(state_dict["scheduler_g"])
        self.scheduler_d.load_state_dict(state_dict["scheduler_d"])

    def forward(self, batch):
        wav = batch["gt_wav"]
        mel = self.mel_generator(wav)
        batch["gen_wav"] = self.generator(mel)
        return batch

    def get_last_lr(self):
        return self.scheduler_g.get_last_lr()[0]

    def scheduler_step(self):
        self.scheduler_g.step()
        self.scheduler_d.step()

    def optimization_step(self, batch, metrics: MetricTracker):
        wav = batch["gt_wav"]
        mel = self.mel_generator(wav)
        gen_wav = self.generator(mel)
        gen_mel = self.mel_generator(gen_wav)
        self.discriminator_step(wav, gen_wav, metrics)
        self.generator_step(wav, mel, gen_wav, gen_mel, metrics)
        batch["gen_wav"] = gen_wav
        return batch

    def freeze_discriminator(self, revert=False):
        for p in self.mp_discriminator.parameters():
            p.requires_grad = revert
        for p in self.ms_discriminator.parameters():
            p.requires_grad = revert

    @staticmethod
    def melspec_loss(real_melspec, fake_melspec):
        return F.l1_loss(fake_melspec, real_melspec)

    @staticmethod
    def feature_loss(real_features, fake_features):
        loss = 0
        for r_features, f_features in zip(real_features, fake_features):
            for r_sub_features, f_sub_features in zip(r_features, f_features):
                loss += F.l1_loss(f_sub_features, r_sub_features)
        return loss

    @staticmethod
    def discriminator_loss(real_score, fake_score):
        loss = 0
        for r_score, f_score in zip(real_score, fake_score):
            loss += torch.mean((1 - r_score) ** 2) + torch.mean(f_score ** 2)
        return loss

    @staticmethod
    def generator_loss(fake_score):
        loss = 0
        for f_score in fake_score:
            loss += torch.mean((1 - f_score) ** 2)
        return loss

    def discriminator_step(self, wav, gen_wav, metrics):
        self.freeze_discriminator(revert=True)

        self.optim_d.zero_grad()

        mp_real_score, _ = self.mp_discriminator(wav)
        mp_fake_score, _ = self.mp_discriminator(gen_wav.detach())
        mp_loss = self.discriminator_loss(mp_real_score, mp_fake_score)

        ms_real_score, _ = self.ms_discriminator(wav)
        ms_fake_score, _ = self.ms_discriminator(gen_wav.detach())
        ms_loss = self.discriminator_loss(ms_real_score, ms_fake_score)

        d_loss = mp_loss + ms_loss

        metrics.update("mp_loss", mp_loss.item())
        metrics.update("ms_loss", ms_loss.item())
        metrics.update("d_loss", d_loss.item())

        d_loss.backward()
        self.optim_d.step()
        

    def generator_step(self, wav, mel, gen_wav, gen_mel, metrics):
        self.freeze_discriminator()

        self.optim_g.zero_grad()

        mel_loss = self.melspec_loss(mel, gen_mel)

        _, mp_real_features = self.mp_discriminator(wav)
        mp_fake_score, mp_fake_features = self.mp_discriminator(gen_wav)

        _, ms_real_features = self.ms_discriminator(wav)
        ms_fake_score, ms_fake_features = self.ms_discriminator(gen_wav)

        mp_feature_loss = self.feature_loss(mp_real_features, mp_fake_features)
        ms_feature_loss = self.feature_loss(ms_real_features, ms_fake_features)

        mp_gen_loss = self.generator_loss(mp_fake_score)
        ms_gen_loss = self.generator_loss(ms_fake_score)

        g_loss = 45 * mel_loss + 2 * (mp_feature_loss + ms_feature_loss) + (mp_gen_loss + ms_gen_loss)

        metrics.update("mel_loss", mel_loss.item())
        metrics.update("mp_feature_loss", mp_feature_loss.item())
        metrics.update("ms_feature_loss", ms_feature_loss.item())
        metrics.update("mp_gen_loss", mp_gen_loss.item())
        metrics.update("ms_gen_loss", ms_gen_loss.item())
        metrics.update("g_loss", g_loss.item())

        g_loss.backward()
        self.optim_g.step()
