import math
import torch
from torch import nn
from torch import Tensor
import gpytorch
from gpytorch.kernels import MultitaskKernel
from gpytorch.constraints import Positive




class TensorProductKernel(MultitaskKernel):
    """
    Class to get the tensorproduct kernel
    """

    def __init__(self, data_covar_module,  num_tasks, rank=1, pos_constraint = None, tri_constaint = None, task_covar_prior=None, **kwargs):
        super().__init__(data_covar_module, num_tasks, rank, task_covar_prior = None, **kwargs)

        
        
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, add_jitter = False, **params):
        if last_dim_is_batch:
            raise RuntimeError("MultitaskKernel does not accept the last_dim_is_batch argument.")
        covar_i = self.task_covar_module.covar_matrix #.evaluate()
            
        covar_i = covar_i.evaluate()
        if len(x1.shape[:-2]):
            covar_i = covar_i.repeat(*x1.shape[:-2], 1, 1)
        covar_x = gpytorch.lazy.lazify(self.data_covar_module.forward(x1, x2, **params))#(self.data_covar_module.forward(x1, x2, **params))#
        if (add_jitter == True):
            covar_x = covar_x #+ (1e-8) * torch.eye(covar_x.shape[0])
        res=gpytorch.lazy.KroneckerProductLazyTensor(covar_x, covar_i) #gpytorch.lazy.lazify(torch.kron(covar_x, covar_i))

        return res.diag() if diag else res
        
    
