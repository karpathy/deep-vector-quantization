"""
Patch Encoders / Decoders as used by DeepMind in their sonnet repo example:
https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
"""
import os

import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, input_channels, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, input_channels, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out += x
        out = F.relu(out)
        return out


import torchvision.models as models

class DeepMindEncoder(nn.Module):

    def __init__(self, input_channels=3, n_hid=64, input_width=32, embedding_dim=64):
        super().__init__()

        strides = [2,2]
        down_sample = np.prod(1/np.array(strides))
        out_width = int(input_width * down_sample)

        if 'SINGLE_TOKEN' in os.environ:
            self.net = nn.Sequential(
                nn.Conv2d(input_channels, n_hid, 4, stride=strides[0], padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_hid, 2*n_hid, 4, stride=strides[1], padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(2*n_hid, 2*n_hid, 3, padding=1),
                nn.ReLU(),
                ResBlock(2*n_hid, 2*n_hid//4),
                ResBlock(2*n_hid, 2*n_hid//4),  # 128, 8x8
                nn.Flatten(),
                nn.Linear(2 * n_hid * out_width ** 2, embedding_dim),
                # TODO: Add to n_embd as well to match expressivity of 8x8 patches in 512 vocab
            )
        else:
            if 'SINGLE_TOKEN2' in os.environ:
                self.output_channels = n_hid  # Want to see what decoding the encoder vs quantized looks like
                self.net = nn.Sequential(
                    nn.Conv2d(input_channels, n_hid, 4, stride=strides[0], padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(n_hid, 2 * n_hid, 4, stride=strides[1], padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(2 * n_hid, self.output_channels, 3, padding=1),
                    nn.ReLU(),
                    ResBlock(self.output_channels, 2 * n_hid // 4),
                    ResBlock(self.output_channels, 2 * n_hid // 4),  # 128, 8x8
                )
            else:
                self.output_channels = 2 * n_hid
                self.net = nn.Sequential(
                    nn.Conv2d(input_channels, n_hid, 4, stride=strides[0], padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(n_hid, 2*n_hid, 4, stride=strides[1], padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(2*n_hid, 2*n_hid, 3, padding=1),
                    nn.ReLU(),
                    ResBlock(2*n_hid, 2*n_hid//4),
                    ResBlock(2*n_hid, 2*n_hid//4),  # 128, 8x8
                )


        self.out_width = out_width
        # self.output_stide = 4

    def forward(self, x):
        return self.net(x)


class DeepMindDecoder(nn.Module):

    def __init__(self, encoder, n_init=32, n_hid=64, output_channels=3, embedding_dim=64):
        super().__init__()

        if 'SINGLE_TOKEN' in os.environ:
            self.net = nn.Sequential(
                nn.Linear(embedding_dim, 2*n_hid * encoder.out_width ** 2),
                nn.Unflatten(1, (2*n_hid, encoder.out_width, encoder.out_width)),
                nn.Conv2d(n_init//encoder.out_width, 2*n_hid, 3, padding=1),
                nn.ReLU(),
                ResBlock(2*n_hid, 2*n_hid//4),
                ResBlock(2*n_hid, 2*n_hid//4),
                nn.ConvTranspose2d(2*n_hid, n_hid, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(n_hid, output_channels, 4, stride=2, padding=1),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(n_init, 2*n_hid, 3, padding=1),
                nn.ReLU(),
                ResBlock(2*n_hid, 2*n_hid//4),
                ResBlock(2*n_hid, 2*n_hid//4),
                nn.ConvTranspose2d(2*n_hid, n_hid, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(n_hid, output_channels, 4, stride=2, padding=1),
            )

    def forward(self, x):
        return self.net(x)
