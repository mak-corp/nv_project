import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class PeriodDiscriminator(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super(PeriodDiscriminator, self).__init__()
        self.period = period
        padding = (get_padding(kernel_size), 0)
        kernel_size = (kernel_size, 1)
        stride = (stride, 1)
        self.convs = nn.ModuleList([
            weight_norm(Conv2d(1, 32, kernel_size, stride, padding)),
            weight_norm(Conv2d(32, 128, kernel_size, stride, padding)),
            weight_norm(Conv2d(128, 512, kernel_size, stride, padding)),
            weight_norm(Conv2d(512, 1024, kernel_size, stride, padding)),
            weight_norm(Conv2d(1024, 1024, kernel_size, stride=1, padding=(2, 0))),
        ])
        self.conv_post = weight_norm(Conv2d(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0)))

    def forward(self, x):
        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = x.shape[-1]
        x = x.view(b, c, t // self.period, self.period)

        features = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        x = torch.flatten(x, start_dim=1)

        return x, features


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period=2),
            PeriodDiscriminator(period=3),
            PeriodDiscriminator(period=5),
            PeriodDiscriminator(period=7),
            PeriodDiscriminator(period=11),
        ])

    def forward(self, x):
        score = []
        features = []
        x = x.view(x.shape[0], 1, -1)
        for discriminator in self.discriminators:
            f_score, f_features = discriminator(x)
            score.append(f_score)
            features.append(f_features)

        return score, features


class ScaleDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(ScaleDiscriminator, self).__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        features = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        x = torch.flatten(x, start_dim=1)

        return x, features


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.pooling = AvgPool1d(4, 2, padding=2)

    def forward(self, x):
        score = []
        features = []
        x = x.view(x.shape[0], 1, -1)
        for i, discriminator in enumerate(self.discriminators):
            f_score, f_features = discriminator(x)
            score.append(f_score)
            features.append(f_features)
            if i + 1 < len(self.discriminators):
                x = self.pooling(x)

        return score, features
