"""
Defines the full (PyTorch Lightning module) VQVAE, which incorporates an
encoder, decoder and a quantize layer in the middle for the discrete bottleneck.
"""

import os
from argparse import ArgumentParser

import torch
from torch import nn, einsum
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data.cifar10 import CIFAR10Data
from model.deepmind_enc_dec import DeepMindEncoder, DeepMindDecoder
from model.quantize import VQVAEQuantize, GumbelQuantize

# -----------------------------------------------------------------------------

class VQVAE(pl.LightningModule):

    def __init__(
        self,
        args,
        in_channel=3, # rgb
        num_hiddens=128, # default deepmind settings
        num_residual_hiddens=32,
        embedding_dim=64,
        num_embeddings=512,
    ):
        super().__init__()

        self.encoder = DeepMindEncoder(in_channel, num_hiddens, num_residual_hiddens)
        self.decoder = DeepMindDecoder(in_channel, num_hiddens, num_residual_hiddens, embedding_dim)

        QuantizerModule = {
            'vqvae': VQVAEQuantize,
            'gumbel': GumbelQuantize,
        }[args.vq_flavor]
        self.quantizer = QuantizerModule(num_hiddens, embedding_dim, num_embeddings)

    def forward(self, x):
        z = self.encoder(x)
        z_q, latent_loss, ind = self.quantizer(z)
        x_hat = self.decoder(z_q)
        return x_hat, latent_loss, ind

    def training_step(self, batch, batch_idx):
        x, y = batch # hate that i have to do this here in the model
        x_hat, latent_loss, ind = self.forward(x)
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        loss = recon_loss + latent_loss
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch # hate that i have to do this here in the model
        x_hat, latent_loss, ind = self.forward(x)

        # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
        encodings = F.one_hot(ind, self.quantizer.n_embed).float().reshape(-1, self.quantizer.n_embed)
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
        cluster_use = torch.sum(avg_probs > 0)
        self.log('val_perplexity', perplexity, prog_bar=True)
        self.log('val_cluster_use', cluster_use, prog_bar=True)

        """
        data variance is fixed, estimated and used by deepmind in their cifar10 example presumably
        to evaluate a proper log probability under a gaussian, except I think they are also
        missing an additional factor of half? Leaving this alone and following their code anyway.
        https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
        """
        data_variance = 0.06327039811675479
        recon_error = F.mse_loss(x_hat, x, reduction='mean') / data_variance
        self.log('val_recon_error', recon_error, prog_bar=True) # DeepMind converges to 0.056 in 4min 29s wallclock

    def configure_optimizers(self):

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 1e-5},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=3e-4, weight_decay=1e-5)

        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--vq_flavor", type=str, default='vqvae', choices=['vqvae', 'gumbel'])
        return parser

def cli_main():
    pl.seed_everything(1337)

    # -------------------------------------------------------------------------
    # arguments...
    parser = ArgumentParser()
    # training related
    parser = pl.Trainer.add_argparse_args(parser)
    # model related
    parser = VQVAE.add_model_specific_args(parser)
    # dataloader related
    parser.add_argument("--data_dir", type=str, default='/apcv/users/akarpathy/cifar10')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    # done!
    args = parser.parse_args()
    # -------------------------------------------------------------------------

    data = CIFAR10Data(args)
    model = VQVAE(args)

    checkpoint_callback = ModelCheckpoint(monitor='val_recon_error', mode='min')

    class DecayTemperature(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            e = trainer.current_epoch
            e0,e1 = 0, 30
            t0,t1 = 1.0, 0.1
            alpha = max(0, min(1, (e - e0) / (e1 - e0)))
            t = alpha * t1 + (1 - alpha) * t0 # probably should be exponential instead
            print("epoch %d setting temperature of model's quantizer to %f" % (e, t))
            pl_module.quantizer.temperature = t

    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback, DecayTemperature()])

    trainer.fit(model, data)

if __name__ == "__main__":
    cli_main()
