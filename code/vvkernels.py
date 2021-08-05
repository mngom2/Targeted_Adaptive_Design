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

    def __init__(self, data_covar_module, chol_11, chol_21, chol_22,  num_tasks, rank=1, pos_constraint = None, tri_constaint = None, task_covar_prior=None, **kwargs):
        super().__init__(data_covar_module, num_tasks, rank, task_covar_prior, **kwargs)
#        self.num_tasks = num_tasks
#        self.rank = rank
#        self.data_covar_module = data_covar_module
#        self.task_covar_prior = task_covar_prior
        
        
        
        # register the raw parameter
        self.register_parameter(
            name='chol_11', parameter=torch.nn.Parameter(chol_11)
        )

        # set the parameter constraint to be triangular
        if pos_constraint is None:
            pos_constraint = Positive()

        # register the constraint
        self.register_constraint("chol_11", pos_constraint)
        
        @property
        def cholesky_11(self):
            return self.chol_11_constraint.transform(self.chol_11)
            
        @cholesky_11.setter
        def cholesky_11(self, value):
            return self._set_cholesky_factor(value)
            
        
        def _set_cholesky_11(self, value):
            if not torch.is_tensor(value):
                value = torch.as_tensor(value).to(self.chol_11)
                
            self.initialize(chol_11 = self.chol_11_constraint.inverse_transform(value))
            
            
            # register the raw parameter
        self.register_parameter(
            name='chol_22', parameter=torch.nn.Parameter(chol_22)
        )


        # register the constraint
        self.register_constraint("chol_22", pos_constraint)
        
        @property
        def cholesky_22(self):
            return self.chol_22_constraint.transform(self.chol_22)
            
        @cholesky_22.setter
        def cholesky_22(self, value):
            return self._set_cholesky_22(value)
            
        
        def _set_cholesky_22(self, value):
            if not torch.is_tensor(value):
                value = torch.as_tensor(value).to(self.chol_22)
                
            self.initialize(chol_22= self.chol_22_constraint.inverse_transform(value))
        
                    # register the raw parameter
        self.register_parameter(
            name='chol_21', parameter=torch.nn.Parameter(chol_21)
        )
        
        @property
        def cholesky_21(self):
            return (self.chol_21)
            
        @cholesky_22.setter
        def cholesky_21(self, value):
            return self._set_cholesky_21(value)
            
        
        def _set_cholesky_21(self, value):
            if not torch.is_tensor(value):
                value = torch.as_tensor(value).to(self.chol_21)
                
            self.initialize(chol_22= (value))

        
        
       # self.chol_q = chol_q

        
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        k = self.data_covar_module
       # chol_factor = self.chol_q
        chol_factor = torch.zeros(self.num_tasks, self.num_tasks)
        chol_factor[0,0] = self.chol_11
        chol_factor[1,1] = self.chol_22
        chol_factor[1,0] = self.chol_21
    
        kappa = gpytorch.matmul(chol_factor, chol_factor.t())
      
        x1 = x1.reshape(x1.shape[0], 1)
        x2 = x2.reshape(x2.shape[0], 1)

        k_fwd = k.forward(x1, x2)
        kronecker_kernel = torch.kron(k_fwd, kappa)
        return kronecker_kernel
#
#        for i in range(self.num_tasks):
#            for j in range(self.num_tasks):
#                k_fwd = k.forward(x1, x2)
#                print(k_fwd.shape)
#                kappa_ij = kappa[i,j]
#                kronecker_kernel[i,j] = torch.kron(k_fwd, kappa_ij)
            
        #print(kronecker_kernel)
              #  kronecker_kernel = torch.zeros(self.num_tasks * d1, self.num_tasks * d2)
#        print(kappa)
         #(kronecker_kernel.reshape(self.num_tasks * d1, self.num_tasks*d2))
        

#        d1 = x1.shape[0]
#        d2 = x2.shape[0]

        
        

    
    
  
