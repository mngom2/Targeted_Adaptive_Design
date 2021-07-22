import math
import torch
from torch import nn
from torch import Tensor
import gpytorch
from gpytorch.kernel import Kernel
from matplotlib import pyplot as plt



class TensorProductKernel(Kernel):
    """
    Class to get the tensorproduct kernel
    """

    def __init__(self, data_covar_module, chol_q,  num_tasks, rank=1, tri_constaint = None, task_covar_prior=None, **kwargs):
        self.num_tasks = num_tasks
        self.rank = rank
        self.data_covar_module = data_covar_module
        self.task_covar_prior = task_covar_prior
        
        # register the raw parameter
        self.register_parameter(
            name='chol_factor', parameter=torch.nn.Parameter(chol_q)
        )

        # set the parameter constraint to be triangular
        if tri_constaint is None:
            tri_constaint = True

        # register the constraint
        self.register_constraint("chol_factor", tri_constraint)
        
        @property
        def cholesky_factor(self):
            return self.chol_factor_constraint.transform(self.chol_factor)
            
        @cholesky_factor.setter
        def cholesky_factor(self, value):
            return self._set_cholesky_factor(value)
            
        
        def _set_cholesky_factor(self, value):
            if not torch.is_tensor(value):
                value = torch.as_tensor(value).to(self.chol_factor)
                
            self.initialize(chol_factor = self.chol_factor_constraint.inverse_transform(value))
        

        
        
       # self.chol_q = chol_q

        
    def forward(self, x1, x2):
        k = self.data_covar_module
       # chol_factor = self.chol_q
        kappa = self.chol_q.mul(self.chol_q.t())
        d1, d2 = self.chol_q.size()
        kronecker_kernel = torch.zeros(self.num_tasks, self.num_tasks, d1, d2)
        
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                k_fwd = k.forward(x1, x2)
                kappa_ij = kappa[i,j]
                kronecker_kernel[i,j] = torch.kron(k_fwd, kappa_ij)
            
        
        return (kronecker_kernel.reshape(self.num_tasks * d1, self.num_tasks*d2))
        



        
        

    
    
  
