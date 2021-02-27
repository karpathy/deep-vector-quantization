import os
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dvq.data.cifar10 import CIFAR10Data
from dvq.model.vqvae import VQVAE

def main():
    pl.seed_everything(1337)

    parser = ArgumentParser()
    # model related
    parser.add_argument("--vq_flavor", type=str, default='vqvae', choices=['vqvae', 'gumbel'])
    # data related
    parser.add_argument("--data_dir", type=str, default='/apcv/users/akarpathy/cifar10')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

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
    main()
