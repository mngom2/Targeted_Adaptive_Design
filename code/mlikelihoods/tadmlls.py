from typing import Any, Optional
import warnings
from copy import deepcopy
import math
import torch
from torch import nn
from torch import Tensor
import gpytorch
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.distributions import base_distributions
from gpytorch.settings import max_cholesky_size
from gpytorch.lazy import ZeroLazyTensor, DiagLazyTensor
from gpytorch.utils.warnings import GPInputWarning
from gpytorch.mlls import ExactMarginalLogLikelihood
from matplotlib import pyplot as plt





#torch.set_default_dtype(torch.float64)


#gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False)
#with torch.no_grad():
gpytorch.settings.fast_computations(covar_root_decomposition = True, log_prob = True, solves = False)
gpytorch.settings.max_cg_iterations(100)
gpytorch.settings.max_preconditioner_size(100)
max_cholesky_size._set_value(3000)
#gpytorch.settings.fast_pred_var(False)


class MarginalLogLikelihood(ExactMarginalLogLikelihood):
    """
    Class to get the mll to be optimized to tune the GP hyperparameter
    """
    
    def __init__(self, likelihood, model):
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference")
        super(ExactMarginalLogLikelihood, self).__init__(likelihood, model)
        
    def forward(self, agg_data, g_theta1,  model, likelihood, cov_noise1):
        r"""
        Computes the MLL given :math:`p(\mathbf f)` and :math:`\mathbf y`.

        :param ~gpytorch.distributions.MultivariateNormal function_dist: :math:`p(\mathbf f)`
            the outputs of the latent function (the :obj:`gpytorch.models.ExactGP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :rtype: torch.Tensor
        :return: Exact MLL. Output shape corresponds to batch shape of the model/input data.
        """
        # move the cov_noise1 to jupyter notebook Conv_Success
#         cov_noise1 =  noise_value * torch.eye(agg_data.shape[0])    #likelihood._shaped_noise_covar([ agg_data.shape[0],

        #output = model(g_theta1)
        #output_ll = likelihood(output)
        m0 = model.mean_module.forward(g_theta1) #output_ll.loc #
        #print(m0)
        #C11= output_ll.covariance_matrix #+ cov_noise1
        K = model.covar_module
 #       mu = model.mean_module.forward(g_theta1)

        C11= K.forward(g_theta1, g_theta1, add_jitter = True) + cov_noise1

        #covar = output_ll.lazy_covariance_matrix
     
        m0 = m0.reshape(agg_data.shape)
     
        diff = agg_data - m0
#        if diff.shape[:-1] != covar.batch_shape:
#            print('error shape')

        inv_quad_C11, logdet_C11 = gpytorch.inv_quad_logdet(C11, diff.unsqueeze(-1), logdet=True)
        pi = Tensor([math.pi])
        N = C11.shape[1]
        
        #val = output_ll.log_prob(agg_data)
        return 1./N * (-.5 * logdet_C11 - .5 * inv_quad_C11 ) , inv_quad_C11 # val # val #
        
        
    
