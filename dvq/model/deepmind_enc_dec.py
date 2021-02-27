"""
Patch Encoders / Decoders as used by DeepMind in their sonnet repo example:
https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
"""

import torch
from torch import nn, einsum
import torch.nn.functional as F

# -----------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out += x
        out = F.relu(out)
        return out


class DeepMindEncoder(nn.Module):

    def __init__(self, in_channel=3, num_hiddens=128, num_residual_hiddens=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channel, num_hiddens//2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_hiddens//2, num_hiddens, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_hiddens, num_hiddens, 3, padding=1),
            nn.ReLU(),
            ResBlock(num_hiddens, num_residual_hiddens),
            ResBlock(num_hiddens, num_residual_hiddens),
        )

    def forward(self, x):
        return self.net(x)


class DeepMindDecoder(nn.Module):

    def __init__(self, in_channel=3, num_hiddens=128, num_residual_hiddens=32, embedding_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(embedding_dim, num_hiddens, 3, padding=1),
            nn.ReLU(),
            ResBlock(num_hiddens, num_residual_hiddens),
            ResBlock(num_hiddens, num_residual_hiddens),
            nn.ConvTranspose2d(num_hiddens, num_hiddens//2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_hiddens//2, in_channel, 4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x)
