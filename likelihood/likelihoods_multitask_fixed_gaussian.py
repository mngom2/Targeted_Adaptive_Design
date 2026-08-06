#!/usr/bin/env python3

import warnings
from typing import Any, Optional

import torch
from torch import Tensor

from gpytorch.constraints import GreaterThan, Interval
from gpytorch.distributions import base_distributions
from gpytorch.functions import add_diag
from gpytorch.lazy import (
    BlockDiagLazyTensor,
    DiagLazyTensor,
    KroneckerProductLazyTensor,
    MatmulLazyTensor,
    RootLazyTensor,
    lazify,
)
from gpytorch.likelihoods import _MultitaskGaussianLikelihoodBase
from gpytorch.priors import Prior
from gpytorch.utils.deprecation import _deprecate_kwarg_with_transform
from .noise_models_multitask_fixed_gaussian import MultitaskFixedGaussianNoise



class MultitaskFixedNoiseGaussianLikelihood(_MultitaskGaussianLikelihoodBase):
    def __init__(self, noise: Tensor, rank: int = 0, task_correlation_prior: Optional[Prior] = None,noise_constraint: Optional[Interval] = None,batch_shape: torch.Size = torch.Size(), **kwargs: Any) -> None:
        """
        Args:
            :attr:`noise` (Tensor):
                `n x t`-dim tensor of observation noise, where `t` is the number of tasks.
            :attr:`rank` (int):
                The rank of the task noise covariance matrix to fit. If 0 (default),
                then a diagonal covariance matrix is fit.
            task_correlation_prior (:obj:`gpytorch.priors.Prior`): Prior to use over the task noise correlaton matrix.
                Only used when `rank` > 0.
        """
#        num_tasks = noise.size(-1)
#        noise_covar = MultitaskFixedGaussianNoise(noise=noise)
        super().__init__(
            num_tasks=noise.size(-1),
            noise_covar=MultitaskFixedGaussianNoise(noise=noise),
            rank=rank,
            task_correlation_prior=task_correlation_prior,
            batch_shape=batch_shape,
        )
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-5)
            
        self.register_parameter(name="raw_noise", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        self.register_constraint("raw_noise", noise_constraint)

    @property
    def noise(self):
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value):
        
        self._set_noise(value)

    def _set_noise(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_noise)
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))
        
        
    def get_fantasy_likelihood(self, **kwargs):
        if "noise" not in kwargs:
            raise RuntimeError("MultitaskFixedNoiseGaussianLikelihood.fantasize requires a `noise` kwarg")
        old_noise_covar = self.noise_covar
        self.noise_covar = None
        fantasy_liklihood = deepcopy(self)
        self.noise_covar = old_noise_covar

        old_noise = old_noise_covar.noise
        new_noise = kwargs.get("noise")
        if old_noise.dim() != new_noise.dim():
            old_noise = old_noise.expand(*new_noise.shape[:-1], old_noise.shape[-1])
        fantasy_liklihood.noise_covar = MultitaskFixedGaussianNoise(noise=torch.cat([old_noise, new_noise], -1))
        return fantasy_liklihood

    def _shaped_noise_covar(self, base_shape, *params: Any, **kwargs: Any):
        #print('kjhg')
        noise_covar = super()._shaped_noise_covar(base_shape, *params, **kwargs)
        noise = self.noise
        #noise = torch.zeros(noise.shape)
        
        #print(noise_covar.add_diag(noise).evaluate())
        return noise_covar.add_diag(noise)


class MultitaskGaussianLikelihoodKronecker(_MultitaskGaussianLikelihoodBase):
    """
    A convenient extension of the :class:`gpytorch.likelihoods.GaussianLikelihood` to the multitask setting that allows
    for a full cross-task covariance structure for the noise. The fitted covariance matrix has rank `rank`.
    If a strictly diagonal task noise covariance matrix is desired, then rank=0 should be set. (This option still
    allows for a different `noise` parameter for each task.)
    Like the Gaussian likelihood, this object can be used with exact inference.
    Note: This Likelihood is scheduled to be deprecated and replaced by an improved version of
    `MultitaskGaussianLikelihood`. Use this only for compatibility with batched Multitask models.
    """

    def __init__(
        self, num_tasks, rank=0, task_prior=None, batch_shape=torch.Size(), noise_prior=None, noise_constraint=None
    ):
        """
        Args:
            num_tasks (int): Number of tasks.
            rank (int): The rank of the task noise covariance matrix to fit. If `rank` is set to 0,
            then a diagonal covariance matrix is fit.
            task_prior (:obj:`gpytorch.priors.Prior`): Prior to use over the task noise covariance matrix if
            `rank` > 0, or a prior over the log of just the diagonal elements, if `rank` == 0.
        """
        super(Likelihood, self).__init__()
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-5)
        self.register_parameter(name="raw_noise", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        if rank == 0:
            self.register_parameter(
                name="raw_task_noises", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, num_tasks))
            )
            if task_prior is not None:
                raise RuntimeError("Cannot set a `task_prior` if rank=0")
        else:
            self.register_parameter(
                name="task_noise_covar_factor", parameter=torch.nn.Parameter(torch.randn(*batch_shape, num_tasks, rank))
            )
            if task_prior is not None:
                self.register_prior("MultitaskErrorCovariancePrior", task_prior, self._eval_covar_matrix)
        self.num_tasks = num_tasks
        self.rank = rank

        self.register_constraint("raw_noise", noise_constraint)

    @property
    def noise(self):
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value):
        self._set_noise(value)

    def _set_noise(self, value):
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    def _eval_covar_matrix(self):
        covar_factor = self.task_noise_covar_factor
        noise = self.noise
        D = noise * torch.eye(self.num_tasks, dtype=noise.dtype, device=noise.device)
        return covar_factor.matmul(covar_factor.transpose(-1, -2)) + D

    def marginal(self, function_dist, *params, **kwargs):
        """
        Adds the task noises to the diagonal of the covariance matrix of the supplied
        :obj:`gpytorch.distributions.MultivariateNormal` or :obj:`gpytorch.distributions.MultitaskMultivariateNormal`,
        in case of `rank` == 0. Otherwise, adds a rank `rank` covariance matrix to it.
        To accomplish this, we form a new :obj:`gpytorch.lazy.KroneckerProductLazyTensor` between :math:`I_{n}`,
        an identity matrix with size equal to the data and a (not necessarily diagonal) matrix containing the task
        noises :math:`D_{t}`.
        We also incorporate a shared `noise` parameter from the base
        :class:`gpytorch.likelihoods.GaussianLikelihood` that we extend.
        The final covariance matrix after this method is then :math:`K + D_{t} \otimes I_{n} + \sigma^{2}I_{nt}`.
        Args:
            function_dist (:obj:`gpytorch.distributions.MultitaskMultivariateNormal`): Random variable whose covariance
                matrix is a :obj:`gpytorch.lazy.LazyTensor` we intend to augment.
        Returns:
            :obj:`gpytorch.distributions.MultitaskMultivariateNormal`: A new random variable whose covariance
            matrix is a :obj:`gpytorch.lazy.LazyTensor` with :math:`D_{t} \otimes I_{n}` and :math:`\sigma^{2}I_{nt}`
            added.
        """
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix

        if self.rank == 0:
            task_noises = self.raw_noise_constraint.transform(self.raw_task_noises)
            task_var_lt = DiagLazyTensor(task_noises)
            dtype, device = task_noises.dtype, task_noises.device
        else:
            task_noise_covar_factor = self.task_noise_covar_factor
            task_var_lt = RootLazyTensor(task_noise_covar_factor)
            dtype, device = (task_noise_covar_factor.dtype, task_noise_covar_factor.device)

        eye_lt = DiagLazyTensor(
            torch.ones(*covar.batch_shape, covar.size(-1) // self.num_tasks, dtype=dtype, device=device)
        )
        task_var_lt = task_var_lt.expand(*covar.batch_shape, *task_var_lt.matrix_shape)

        covar_kron_lt = KroneckerProductLazyTensor(eye_lt, task_var_lt)
        covar = covar + covar_kron_lt

        noise = self.noise
        covar = add_diag(covar, noise)
        return function_dist.__class__(mean, covar)
