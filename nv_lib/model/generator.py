import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm


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
        upsample_in_channels = 512  # 128
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
