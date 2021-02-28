"""
VQVAE losses: e.g. L1, L2, "Logit Laplace" from DALL-E work, etc.
"""

import torch

# -----------------------------------------------------------------------------

logit_laplace_eps: float = 0.1

def map_pixels(x: torch.Tensor) -> torch.Tensor:
    """ map [0,1] range to [eps, 1-eps] """
    if x.dtype != torch.float:
        raise ValueError('expected input to have type float')
    return (1 - 2 * logit_laplace_eps) * x + logit_laplace_eps


def unmap_pixels(x: torch.Tensor) -> torch.Tensor:
    """ inverse map, from [eps, 1-eps] to [0,1], with clamping """
    if x.dtype != torch.float:
        raise ValueError('expected input to have type float')
    return torch.clamp((x - logit_laplace_eps) / (1 - 2 * logit_laplace_eps), 0, 1)
