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

    def __init__(self, data_covar_module, chol_q,  num_tasks, rank=1, pos_constraint = None, tri_constaint = None, task_covar_prior=None, **kwargs):
        super().__init__(data_covar_module, num_tasks, rank, task_covar_prior, **kwargs)
#        self.num_tasks = num_tasks
#        self.rank = rank
#        self.data_covar_module = data_covar_module
#        self.task_covar_prior = task_covar_prior
        
        
        
        # register the raw parameter
        self.register_parameter(
            name='chol_factor', parameter=torch.nn.Parameter(chol_q)
        )

        # set the parameter constraint to be triangular
        if pos_constraint is None:
            pos_constraint = Positive()

        # register the constraint
        self.register_constraint("chol_factor", pos_constraint)
        
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

        
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        k = self.data_covar_module
       # chol_factor = self.chol_q
        kappa = self.chol_factor.mul(self.chol_factor.t())
        x1 = x1.reshape(x1.shape[0], 1)
        x2 = x2.reshape(x2.shape[0], 1)
        d1 = x1.shape[0]
        d2 = x2.shape[0]
        kronecker_kernel = torch.zeros(self.num_tasks * d1, self.num_tasks*d2)
        k_fwd = k.forward(x1, x2)
        kronecker_kernel = torch.kron(k_fwd, kappa)
#
#        for i in range(self.num_tasks):
#            for j in range(self.num_tasks):
#                k_fwd = k.forward(x1, x2)
#                print(k_fwd.shape)
#                kappa_ij = kappa[i,j]
#                kronecker_kernel[i,j] = torch.kron(k_fwd, kappa_ij)
            
        
        return kronecker_kernel #(kronecker_kernel.reshape(self.num_tasks * d1, self.num_tasks*d2))
        



        
        

    
    
  
