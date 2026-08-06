#!/usr/bin/env python3

from typing import Any, Optional
import torch
from torch import Tensor
from torch.nn import Parameter

import gpytorch
from gpytorch.likelihoods.noise_models import FixedGaussianNoise
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import ConstantDiagLazyTensor, DiagLazyTensor, ZeroLazyTensor




class MultitaskFixedGaussianNoise(FixedGaussianNoise):
    """Fixed Gaussian observation noise for multi-task models.
    To be used with FixedNoiseGaussianLikelihood.
    Args:
        :attr:`noise` (Tensor):
            `n x t`-dim tensor of observation noise, where `t` is the number of tasks.
    """

    def __init__(self, noise: Tensor) -> None:
        if not noise.ndim > 1:
            raise ValueError("noise must be at least two-dimensional for MultitaskFixedGaussianNoise")
        super().__init__(noise=noise)

    def forward(
        self, *params: Any, shape: Optional[torch.Size] = None, noise: Optional[Tensor] = None, **kwargs: Any
    ) -> DiagLazyTensor:
        if shape is None:
            p = params[0] if torch.is_tensor(params[0]) else params[0][0]
            shape = p.shape if len(p.shape) == 1 else p.shape[:-1]

        if noise is not None:
            return DiagLazyTensor(noise)
    
        elif shape[-1] == self.noise.shape[-2]:
            return DiagLazyTensor(self.noise)
        else:
            return ZeroLazyTensor()


def clear_cache_hook(module, grad_input, grad_output) -> None:
    try:
        module.noise_model.prediction_strategy = None
    except AttributeError:
        pass
