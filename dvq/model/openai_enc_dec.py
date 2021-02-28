"""
OpenAI DALL-E Encoder/Decoder, taken and modified from their official repo @
https://github.com/openai/DALL-E

- Removed first/last 1x1 convs because in this repo those are part of the Quantize layers. This
  is done so that VQVAE and GumbelSoftmax can be viewed side by side cleaner and more symmetrically.
- Got rid of some of the fp16 / device / requires_grad settings, we're going to keep things simple
"""

import attr
import math
from collections import OrderedDict
from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------

@attr.s(eq=False)
class Conv2d(nn.Module): # TODO: simplify to standard PyTorch Conv2d
    n_in:  int = attr.ib(validator=lambda i, a, x: x >= 1)
    n_out: int = attr.ib(validator=lambda i, a, x: x >= 1)
    kw:    int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 2 == 1)

    def __attrs_post_init__(self) -> None:
        super().__init__()

        w = torch.empty((self.n_out, self.n_in, self.kw, self.kw), dtype=torch.float32)
        w.data.normal_(std=1/math.sqrt(self.n_in * self.kw ** 2))

        b = torch.zeros((self.n_out,), dtype=torch.float32)

        self.weight, self.bias = nn.Parameter(w), nn.Parameter(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, self.bias, padding=(self.kw - 1) // 2)


@attr.s(eq=False, repr=False)
class EncoderBlock(nn.Module):
    n_in:     int = attr.ib(validator=lambda i, a, x: x >= 1)
    n_out:    int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 4 ==0)
    n_layers: int = attr.ib(validator=lambda i, a, x: x >= 1)

    def __attrs_post_init__(self) -> None:
        super().__init__()
        self.n_hid = self.n_out // 4
        self.post_gain = 1 / (self.n_layers ** 2)

        make_conv     = partial(Conv2d)
        self.id_path  = make_conv(self.n_in, self.n_out, 1) if self.n_in != self.n_out else nn.Identity()
        self.res_path = nn.Sequential(OrderedDict([
                ('relu_1', nn.ReLU()),
                ('conv_1', make_conv(self.n_in,  self.n_hid, 3)),
                ('relu_2', nn.ReLU()),
                ('conv_2', make_conv(self.n_hid, self.n_hid, 3)),
                ('relu_3', nn.ReLU()),
                ('conv_3', make_conv(self.n_hid, self.n_hid, 3)),
                ('relu_4', nn.ReLU()),
                ('conv_4', make_conv(self.n_hid, self.n_out, 1)),]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)

@attr.s(eq=False, repr=False)
class OpenAIEncoder(nn.Module):
    input_channels:  int = attr.ib(default=3,    validator=lambda i, a, x: x >= 1)
    n_hid:           int = attr.ib(default=256,  validator=lambda i, a, x: x >= 64)
    n_blk_per_group: int = attr.ib(default=2,    validator=lambda i, a, x: x >= 1)

    def __attrs_post_init__(self) -> None:
        super().__init__()

        group_count = 4
        blk_range  = range(self.n_blk_per_group)
        n_layers   = group_count * self.n_blk_per_group
        make_conv  = partial(Conv2d)
        make_blk   = partial(EncoderBlock, n_layers=n_layers)

        self.blocks = nn.Sequential(OrderedDict([
            ('input', make_conv(self.input_channels, 1 * self.n_hid, 7)),
            ('group_1', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(1 * self.n_hid, 1 * self.n_hid)) for i in blk_range],
                ('pool', nn.MaxPool2d(kernel_size=2)),
            ]))),
            ('group_2', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(1 * self.n_hid if i == 0 else 2 * self.n_hid, 2 * self.n_hid)) for i in blk_range],
                ('pool', nn.MaxPool2d(kernel_size=2)),
            ]))),
            ('group_3', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(2 * self.n_hid if i == 0 else 4 * self.n_hid, 4 * self.n_hid)) for i in blk_range],
                ('pool', nn.MaxPool2d(kernel_size=2)),
            ]))),
            ('group_4', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(4 * self.n_hid if i == 0 else 8 * self.n_hid, 8 * self.n_hid)) for i in blk_range],
            ]))),
            ('output', nn.Sequential(OrderedDict([
                ('relu', nn.ReLU()),
            ]))),
        ]))

        self.output_channels = 8 * self.n_hid
        self.output_stide = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 4:
            raise ValueError(f'input shape {x.shape} is not 4d')
        if x.shape[1] != self.input_channels:
            raise ValueError(f'input has {x.shape[1]} channels but model built for {self.input_channels}')
        if x.dtype != torch.float32:
            raise ValueError('input must have dtype torch.float32')

        return self.blocks(x)


@attr.s(eq=False, repr=False)
class DecoderBlock(nn.Module):
    n_in:     int = attr.ib(validator=lambda i, a, x: x >= 1)
    n_out:    int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 4 ==0)
    n_layers: int = attr.ib(validator=lambda i, a, x: x >= 1)

    def __attrs_post_init__(self) -> None:
        super().__init__()
        self.n_hid = self.n_out // 4
        self.post_gain = 1 / (self.n_layers ** 2)

        make_conv     = partial(Conv2d)
        self.id_path  = make_conv(self.n_in, self.n_out, 1) if self.n_in != self.n_out else nn.Identity()
        self.res_path = nn.Sequential(OrderedDict([
                ('relu_1', nn.ReLU()),
                ('conv_1', make_conv(self.n_in,  self.n_hid, 1)),
                ('relu_2', nn.ReLU()),
                ('conv_2', make_conv(self.n_hid, self.n_hid, 3)),
                ('relu_3', nn.ReLU()),
                ('conv_3', make_conv(self.n_hid, self.n_hid, 3)),
                ('relu_4', nn.ReLU()),
                ('conv_4', make_conv(self.n_hid, self.n_out, 3)),]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)

@attr.s(eq=False, repr=False)
class OpenAIDecoder(nn.Module):
    n_init:          int = attr.ib(default=128,  validator=lambda i, a, x: x >= 8)
    n_hid:           int = attr.ib(default=256,  validator=lambda i, a, x: x >= 64)
    output_channels: int = attr.ib(default=3,    validator=lambda i, a, x: x >= 1)
    n_blk_per_group: int = attr.ib(default=2,    validator=lambda i, a, x: x >= 1)

    def __attrs_post_init__(self) -> None:
        super().__init__()

        group_count = 4
        blk_range  = range(self.n_blk_per_group)
        n_layers   = group_count * self.n_blk_per_group
        make_conv  = partial(Conv2d)
        make_blk   = partial(DecoderBlock, n_layers=n_layers)

        self.blocks = nn.Sequential(OrderedDict([
            ('group_1', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(self.n_init if i == 0 else 8 * self.n_hid, 8 * self.n_hid)) for i in blk_range],
                ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
            ]))),
            ('group_2', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(8 * self.n_hid if i == 0 else 4 * self.n_hid, 4 * self.n_hid)) for i in blk_range],
                ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
            ]))),
            ('group_3', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(4 * self.n_hid if i == 0 else 2 * self.n_hid, 2 * self.n_hid)) for i in blk_range],
                ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
            ]))),
            ('group_4', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(2 * self.n_hid if i == 0 else 1 * self.n_hid, 1 * self.n_hid)) for i in blk_range],
            ]))),
            ('output', nn.Sequential(OrderedDict([
                ('relu', nn.ReLU()),
                ('conv', make_conv(1 * self.n_hid, self.output_channels, 1)),
            ]))),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 4:
            raise ValueError(f'input shape {x.shape} is not 4d')
        if x.dtype != torch.float32:
            raise ValueError('input must have dtype torch.float32')

        return self.blocks(x)