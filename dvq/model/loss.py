"""
VQVAE losses, used for the reconstruction term in the ELBO
"""

import math
import torch

# -----------------------------------------------------------------------------

class LogitLaplace:
    """ the Logit Laplace distribution log likelihood from OpenAI's DALL-E paper """
    logit_laplace_eps = 0.1

    class InMap:
        def __call__(self, x):
            # map [0,1] range to [eps, 1-eps]
            return (1 - 2 * LogitLaplace.logit_laplace_eps) * x + LogitLaplace.logit_laplace_eps

    @classmethod
    def unmap(cls, x):
        # inverse map, from [eps, 1-eps] to [0,1], with clamping
        return torch.clamp((x - cls.logit_laplace_eps) / (1 - 2 * cls.logit_laplace_eps), 0, 1)

    @classmethod
    def nll(cls, x, mu_logb):
        raise NotImplementedError # coming right up


class Normal:
    """
    simple normal distribution with fixed variance, as used by DeepMind in their VQVAE
    note that DeepMind's reconstruction loss (I think incorrectly?) misses a factor of 2,
    which I have added to the normalizer of the reconstruction loss in nll(), we'll report
    number that is half of what we expect in their jupyter notebook
    """
    data_variance = 0.06327039811675479 # cifar-10 data variance, from deepmind sonnet code
    mean = 0.5

    class InMap:
        def __call__(self, x):
            # these will put numbers into range [-0.5, 0.5],
            #  as used by DeepMind in their sonnet VQVAE example
            return x - Normal.mean # map [0,1] range to [-0.5, 0.5]

    @classmethod
    def unmap(cls, x):
        return torch.clamp(x + Normal.mean, 0, 1)

    @classmethod
    def nll(cls, x, mu):
        return ((x - mu)**2).mean() / (2 * cls.data_variance) #+ math.log(math.sqrt(2 * math.pi * cls.data_variance))
