import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

LRELU_SLOPE = 0.1


def init_conv_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ConvBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.LeakyReLU(0.1),
            weight_norm(Conv1d(channels, channels, kernel_size, stride=1, dilation=dilation,
                padding=get_padding(kernel_size, dilation))),
            nn.LeakyReLU(0.1),
            weight_norm(Conv1d(channels, channels, kernel_size, stride=1, dilation=1,
                padding=get_padding(kernel_size, 1))),
        )
        self.layers.apply(init_conv_weights)

    def forward(self, x):
        return self.layers(x)


class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.convs = nn.ModuleList([ConvBlock(channels, kernel_size, dilation) for dilation in dilations])

    def forward(self, x):
        for conv in self.convs:
            x = x + conv(x)
        return x


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, upsample_rate, upsample_kernel, res_kernels, res_dilations):
        super(UpsamplingBlock, self).__init__()
        pad = (upsample_kernel - upsample_rate) // 2
        self.up_conv = weight_norm(ConvTranspose1d(
            in_channels, in_channels // 2, upsample_kernel, upsample_rate, padding=pad))
        self.up_conv.apply(init_conv_weights)

        self.residuals = nn.ModuleList()
        for kernel, dilation in zip(res_kernels, res_dilations):
            self.residuals.append(ResBlock(in_channels // 2, kernel, dilation))

    def forward(self, x):
        x = F.leaky_relu(x, 0.1)
        x = self.up_conv(x)
        sum_x = None
        for residual in self.residuals:
            out = residual(x)
            sum_x = out if sum_x is None else sum_x + out
        x = sum_x / len(self.residuals)
        return x


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        resblock_kernel_sizes = [3,7,11]
        resblock_dilation_sizes = [[1,3,5], [1,3,5], [1,3,5]]
        upsample_in_channels = 128
        upsample_rates = [8,8,2,2]
        upsample_kernel_sizes = [16,16,4,4]

        self.conv_pre = weight_norm(Conv1d(in_channels=80, out_channels=upsample_in_channels, kernel_size=7, stride=1, padding=3))

        self.upsamplings = nn.ModuleList()
        cur_in_channels = upsample_in_channels
        for upsample_rate, upsample_kernel in zip(upsample_rates, upsample_kernel_sizes):
            up = UpsamplingBlock(cur_in_channels, upsample_rate, upsample_kernel, resblock_kernel_sizes, resblock_dilation_sizes)
            self.upsamplings.append(up)
            cur_in_channels //= 2

        self.conv_post = weight_norm(Conv1d(cur_in_channels, out_channels=1, kernel_size=7, stride=1, padding=3))
        self.conv_post.apply(init_conv_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for upsampling in self.upsamplings:
            x = upsampling(x)
        x = F.leaky_relu(x, 0.01)
        x = self.conv_post(x)
        x = torch.tanh(x)
        x = x.squeeze(1)

        return x


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
        self.meanpool = nn.Identity() if use_spectral_norm else AvgPool1d(4, 2, padding=2)
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
        x = self.meanpool(x)
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

    def forward(self, x):
        score = []
        features = []
        x = x.view(x.shape[0], 1, -1)
        for discriminator in self.discriminators:
            f_score, f_features = discriminator(x)
            score.append(f_score)
            features.append(f_features)

        return score, features
