"""
The critical quantization layers that we sandwich in the middle of the autoencoder
(between the encoder and decoder) that force the representation through a categorical
variable bottleneck and use various tricks (softening / straight-through estimators)
to backpropagate through the sampling process.
"""
import os

import torch
from torch import nn, einsum
import torch.nn.functional as F

from scipy.cluster.vq import kmeans2

# -----------------------------------------------------------------------------

class VQVAEQuantize(nn.Module):
    """
    Neural Discrete Representation Learning, van den Oord et al. 2017
    https://arxiv.org/abs/1711.00937

    Follows the original DeepMind implementation
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
    """
    def __init__(self, num_hiddens, n_embed, embedding_dim, patch_width=None):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.kld_scale = 10.0

        self.output_proj = embedding_dim

        if 'SINGLE_TOKEN' in os.environ:
            self.proj = nn.Linear(embedding_dim, embedding_dim)  # Perhaps could be removed
        else:
            if 'SINGLE_TOKEN2' in os.environ:
                self.output_proj = embedding_dim // patch_width ** 2
            self.proj = nn.Conv2d(num_hiddens, self.output_proj, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.register_buffer('data_initialized', torch.zeros(1))

        self.data_init_buffer = []
        self.data_init_points = 0

    def forward(self, z):
        if 'SINGLE_TOKEN' in os.environ:
            B, E = z.size()  # B, Embed dim
            z_e = self.proj(z)
            flatten = z_e
        else:
            B, C, H, W = z.size()
            z_e = self.proj(z)  #  (B, E, H, W)  # Output proj channels = E
            z_e = z_e.permute(0, 2, 3, 1)  # make (B, H, W, E)  128, 8, 8, 64
            if 'SINGLE_TOKEN2' in os.environ:
                # Enlarge token size (embedding_dim) so that we get one image per token,
                # instead of a grid of image patch tokens
                z_e = z_e.reshape(B, self.embedding_dim)  # B * H * W, E => B, H * W * E
                flatten = z_e
            else:
                flatten = z_e.reshape(-1, self.embedding_dim)  # 8192 (128*8*8), 64  and flatten out space, so (B, E, H, W) -> (B*H*W, E) - a bunch of embeddings

        # DeepMind def does not do this but I find I have to... ;/
        # Works just as well with one point per cluster in single token regime which is somewhat sus.
        if self.training and self.data_initialized.item() == 0:
            # TODO: Build up a larger batch (at least greater than self.n_embed) orig had 64B = n_embd
            print(f'kmeans batch {round(self.data_init_points/(self.n_embed * 64) * 100)}%')
            if self.data_init_points < self.n_embed * 64:  # Let's ensure 64 points per cluster like Karpathy originally had
                self.data_init_buffer.append(flatten)
                self.data_init_points += flatten.size(0)
            else:
                # Stack data inits into tensor
                print('running kmeans!!') # data driven initialization for the embeddings
                init_data = torch.cat(self.data_init_buffer, dim=0)
                # rp = torch.randperm(init_data.size(0))
                kd = kmeans2(init_data.data.cpu().numpy(), self.n_embed, minit='points')  # flatten: 512,1024 vs 8192,64
                # kd = kmeans2(init_data[rp[:20000]].data.cpu().numpy(), self.n_embed, minit='points')  # flatten: 512,1024 vs 8192,64
                self.embed.weight.data.copy_(torch.from_numpy(kd[0]))
                self.data_init_buffer.clear()
                self.data_initialized.fill_(1)
            # TODO: this won't work in multi-GPU setups

        # Extract indexes from embedding and computes distance (similar to k-means here?)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1, keepdim=True).t()
        )
        # Patches initial 8192, 512
        # tensor([[82.1937, 63.8440, 91.1361, ..., 57.0266, 63.1145, 73.9383],
        #         [82.0183, 64.1438, 91.2648, ..., 57.1167, 63.2859, 74.1988],
        #         [82.0561, 64.2067, 91.2553, ..., 57.1494, 63.2686, 74.2414],
        #         ...,
        #         [82.4529, 63.6202, 91.5170, ..., 57.1837, 63.1078, 74.2306],
        #         [82.3836, 63.6463, 91.3414, ..., 57.2186, 63.2163, 74.1406],
        #         [82.2321, 63.5611, 91.3235, ..., 57.2505, 63.3168, 74.1962]],
        #        device='cuda:0')
        # Single token initial 128, 512
        # tensor([[1070.1558, 1015.0119, 1036.4126,  ..., 1010.0697, 1042.3407,
        #          1017.6580],
        #         [1070.6438, 1015.4285, 1036.7009,  ..., 1009.8075, 1042.2577,
        #          1018.0328],
        #         [1070.6200, 1015.2269, 1036.4867,  ..., 1010.4182, 1042.1997,
        #          1018.2223],
        #         ...,
        #         [1070.5081, 1015.1932, 1036.2366,  ..., 1009.8676, 1042.4606,
        #          1017.7983],
        #         [1070.3368, 1015.2463, 1036.3898,  ..., 1009.9712, 1042.3112,
        #          1017.6896],
        #         [1070.3322, 1015.4224, 1036.3407,  ..., 1010.2289, 1042.0529,
        #          1018.1196]], device='cuda:0')

        _, ind = (-dist).max(1)
        if 'SINGLE_TOKEN' not in os.environ and 'SINGLE_TOKEN2' not in os.environ:
            # tensor([[[371, 371, 371,  ..., 371, 371, 371],
            #          [371, 371, 371,  ..., 371, 371, 371],
            #          [371, 371, 371,  ..., 371, 371, 371]]], device='cuda:0')
            ind = ind.view(B, H, W)  # (128, 8, 8)
        # Single token initial 128
        # tensor([411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411,
        #         411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411,
        #         411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411,
        #         411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411,
        #         411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411,
        #         411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411,
        #         411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411,
        #         411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411,
        #         411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411,
        #         411, 411], device='cuda:0')

        # vector quantization cost that trains the embedding vectors
        z_q = self.embed_code(ind) # (B, H, W, C) (128, 8, 8, 64) OR ST2=> (B, E) (128, 4096)
        commitment_cost = 0.25
        latent_loss = commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q - z_e.detach()).pow(2).mean()
        latent_loss *= self.kld_scale

        z_q = z_e + (z_q - z_e).detach() # noop in forward pass, straight-through gradient estimator in backward pass
        if 'SINGLE_TOKEN2' in os.environ:
            # Had 128 * 64 = B * W **2, E
            # Now we have B, W ** 2 * C = 128,
            z_q = z_q.reshape(B, H, W, self.output_proj)  # (B, E) = (B, H*W*C) => (B, H, W, C)
        if 'SINGLE_TOKEN' not in os.environ:
            z_q = z_q.permute(0, 3, 1, 2) # stack encodings into channels again: (B, C, H, W)

        return z_q, latent_loss, ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)


class GumbelQuantize(nn.Module):
    """
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, num_hiddens, n_embed, embedding_dim, straight_through=False):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = 1.0
        self.kld_scale = 5e-4

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

    def forward(self, z):

        # force hard = True when we are in eval mode, as we must quantize
        hard = self.straight_through if self.training else True

        logits = self.proj(z)
        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard)
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kld_scale * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)
        return z_q, diff, ind
