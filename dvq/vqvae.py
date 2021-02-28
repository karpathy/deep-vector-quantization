"""
Defines the full (PyTorch Lightning module) VQVAE, which incorporates an
encoder, decoder and a quantize layer in the middle for the discrete bottleneck.
"""

import os
import math
from argparse import ArgumentParser

import torch
from torch import nn, einsum
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data.cifar10 import CIFAR10Data
from model.deepmind_enc_dec import DeepMindEncoder, DeepMindDecoder
from model.openai_enc_dec import OpenAIEncoder, OpenAIDecoder
from model.openai_enc_dec import Conv2d as PatchedConv2d
from model.quantize import VQVAEQuantize, GumbelQuantize
from model.loss import Normal, LogitLaplace

# -----------------------------------------------------------------------------

class VQVAE(pl.LightningModule):

    def __init__(
        self,
        args,
        input_channels=3,   # number of channels at the input (for images typically 3 = RGB)
        n_hid=64,           # controls the size of the model
        embedding_dim=32,   # the embedding size for each vocabulary element
        num_embeddings=512, # size of the vocabulary at the middle
    ):
        super().__init__()

        # encoder/decoder module pair
        Encoder, Decoder = {
            'deepmind': (DeepMindEncoder, DeepMindDecoder),
            'openai': (OpenAIEncoder, OpenAIDecoder),
        }[args.enc_dec_flavor]
        self.encoder = Encoder(input_channels=input_channels, n_hid=n_hid)
        self.decoder = Decoder(n_init=embedding_dim, n_hid=n_hid, output_channels=input_channels)

        # the quantizer module sandwiched between them, +contributes a KL(posterior || prior) loss to ELBO
        QuantizerModule = {
            'vqvae': VQVAEQuantize,
            'gumbel': GumbelQuantize,
        }[args.vq_flavor]
        self.quantizer = QuantizerModule(self.encoder.output_channels, num_embeddings, embedding_dim)

        # the data reconstruction loss in the ELBO
        ReconLoss = {
            'l2': Normal,
            'logit_laplace': LogitLaplace,
            # todo: add vqgan
        }[args.loss_flavor]
        self.recon_loss = ReconLoss

    def forward(self, x):
        z = self.encoder(x)
        z_q, latent_loss, ind = self.quantizer(z)
        x_hat = self.decoder(z_q)
        return x_hat, latent_loss, ind

    def training_step(self, batch, batch_idx):
        x, y = batch # hate that i have to do this here in the model
        x = self.recon_loss.inmap(x)
        x_hat, latent_loss, ind = self.forward(x)
        recon_loss = self.recon_loss.nll(x, x_hat).mean()
        loss = recon_loss + latent_loss
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch # hate that i have to do this here in the model
        x = self.recon_loss.inmap(x)
        x_hat, latent_loss, ind = self.forward(x)
        recon_loss = self.recon_loss.nll(x, x_hat).mean()
        self.log('val_recon_loss', recon_loss, prog_bar=True)

        # debugging: cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
        encodings = F.one_hot(ind, self.quantizer.n_embed).float().reshape(-1, self.quantizer.n_embed)
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
        cluster_use = torch.sum(avg_probs > 0)
        self.log('val_perplexity', perplexity, prog_bar=True)
        self.log('val_cluster_use', cluster_use, prog_bar=True)

    def configure_optimizers(self):

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d, PatchedConv2d)
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
        optimizer = torch.optim.AdamW(optim_groups, lr=3e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        self.optimizer = optimizer

        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--vq_flavor", type=str, default='vqvae', choices=['vqvae', 'gumbel'])
        parser.add_argument("--enc_dec_flavor", type=str, default='deepmind', choices=['deepmind', 'openai'])
        parser.add_argument("--loss_flavor", type=str, default='l2', choices=['l2', 'logit_laplace'])
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

    # annealing schedules for lots of constants
    checkpoint_callback = ModelCheckpoint(monitor='val_recon_loss', mode='min')

    def cos_anneal(e0, e1, t0, t1, e):
        """ ramp from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1] """
        alpha = max(0, min(1, (e - e0) / (e1 - e0))) # what fraction of the way through are we
        alpha = 1.0 - math.cos(alpha * math.pi/2) # warp through cosine
        t = alpha * t1 + (1 - alpha) * t0 # interpolate accordingly
        return t

    # these follow the OpenAI DALL-E paper recommendations *very roughly*
    class DecayTemperature(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            t = cos_anneal(0, 50, 1.0, 1.0/16, trainer.current_epoch)
            print("epoch %d: setting temperature of model's quantizer to %f" % (trainer.current_epoch, t))
            pl_module.quantizer.temperature = t

    class RampBeta(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            t = cos_anneal(0, 20, 0.0, 5e-4, trainer.current_epoch)
            print("epoch %d: setting kld scale to %e" % (trainer.current_epoch, t))
            pl_module.quantizer.kld_scale = t

    class DecayLR(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            t = cos_anneal(0, 190, 3e-4, 1e-5, trainer.current_epoch)
            print("epoch %d: setting learning rate to %e" % (trainer.current_epoch, t))
            for g in pl_module.optimizer.param_groups:
                g['lr'] = t

    callbacks = [checkpoint_callback, DecayLR()]
    if args.vq_flavor == 'gumbel':
        callbacks.extend([DecayTemperature(), RampBeta()])
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, max_epochs=200)

    trainer.fit(model, data)

if __name__ == "__main__":
    cli_main()
