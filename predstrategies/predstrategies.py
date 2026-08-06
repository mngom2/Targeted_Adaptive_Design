#!/usr/bin/env python3

import functools
import string

import torch
import gpytorch
from gpytorch.settings import max_cholesky_size
from gpytorch.lazy import ZeroLazyTensor, DiagLazyTensor
from gpytorch.utils.warnings import GPInputWarning

gpytorch.settings.fast_computations(covar_root_decomposition = True, log_prob = True, solves = False)
gpytorch.settings.max_cg_iterations(100)
gpytorch.settings.max_preconditioner_size(100)
max_cholesky_size._set_value(3000)


"""
Class to get predictive mean and predictive covariance (ss Rasmussen's book)
"""
class GPprediction(object):
    def __init__(self,model):
        self.cov = model.covar_module
        self.mean = model.mean_module

    def GPpred(self, train_inputs, agg_data, test_x, noise_value):
        n = train_inputs.shape[0]
        C11 = self.cov(train_inputs, train_inputs) + torch.eye(2*n) * noise_value
        Cx1 = self.cov(test_x, train_inputs)
        f_ = C11.inv_matmul(agg_data, Cx1.evaluate())
        C1x = Cx1.t()
        Cxx = self.cov(test_x, test_x)
        cov = Cxx - C11.inv_matmul(C1x.evaluate(), Cx1.evaluate())

        return f_, cov


