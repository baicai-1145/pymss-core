import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d
from torch.nn.utils import remove_weight_norm, weight_norm
from .resample import UpSample1d, DownSample1d
from .commons import init_weights, get_padding
from .activations import SnakeBeta

class Activation1d(nn.Module):
    def __init__(self,
                 activation,
                 up_ratio: int = 2,
                 down_ratio: int = 2,
                 up_kernel_size: int = 12,
                 down_kernel_size: int = 12):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    # x: [B,C,T]
    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)

        return x

class AMPBlock0(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), activation=None):
        super(AMPBlock0, self).__init__()

        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2]))),
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
        ])
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2) # total number of conv layers


        self.activations = nn.ModuleList([
            Activation1d(
                activation=SnakeBeta(channels, alpha_logscale=True))
                for _ in range(self.num_layers)
        ])

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))
        resblock = AMPBlock0

        self.resblocks = nn.ModuleList()
        for i in range(1):
            ch = upsample_initial_channel//(2**(i))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d, activation="snakebeta"))

        activation_post = SnakeBeta(ch, alpha_logscale=True)
        self.activation_post = Activation1d(activation=activation_post)

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        x = F.interpolate(x, int(x.shape[-1] * 3), mode='linear')

        xs = self.resblocks[2](x)
       # xs += self.resblocks[1](x) removing this but adding activation post is similar quality but almost 30% faster
        xs += self.resblocks[0](x)
        xs = xs/2
        
        xs = self.activation_post(xs)
        x = self.conv_post(xs)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.resblocks:
            l.remove_weight_norm()

class SynthesizerTrn(nn.Module):
    """FlashSR upsampler generator (HierSpeech++ architecture)."""
    def __init__(self, spec_channels, segment_size, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes):
        super().__init__()
        self.spec_channels = spec_channels
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.dec = Generator(1, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes)

    def forward(self, x):
        return self.dec(x)

    @torch.no_grad()
    def infer(self, x, max_len=None):
        return self.dec(x[:, :, :max_len])

class FlashSR(SynthesizerTrn):
    """FlashSR super-resolution wrapper exposing the pymss model interface."""
    DEFAULT_CONFIG = {
        "spec_channels": 128,
        "segment_size": 30,
        "resblock": "0",
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "upsample_rates": [3],
        "upsample_initial_channel": 32,
        "upsample_kernel_sizes": [3],
    }

    def __init__(self, spec_channels=None, segment_size=None, resblock=None, resblock_kernel_sizes=None, resblock_dilation_sizes=None, upsample_rates=None, upsample_initial_channel=None, upsample_kernel_sizes=None):
        params = dict(self.DEFAULT_CONFIG)
        overrides = {
            "spec_channels": spec_channels,
            "segment_size": segment_size,
            "resblock": resblock,
            "resblock_kernel_sizes": resblock_kernel_sizes,
            "resblock_dilation_sizes": resblock_dilation_sizes,
            "upsample_rates": upsample_rates,
            "upsample_initial_channel": upsample_initial_channel,
            "upsample_kernel_sizes": upsample_kernel_sizes,
        }
        params.update({k: v for k, v in overrides.items() if v is not None})
        super().__init__(**params)
        self.audio_channels = 1
        self.sample_rate = 16000
        self.output_sample_rate = 48000
