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
        super().__init__(data_covar_module, num_tasks, rank, task_covar_prior = None, **kwargs)
#        self.num_tasks = num_tasks
#        self.rank = rank
#        self.data_covar_module = data_covar_module
#        self.task_covar_prior = task_covar_prior
        
        #self.task_covar_module = None
        
       

        # set the parameter constraint to be triangular
        if pos_constraint is None:
            pos_constraint = Positive()
            
        
                # register the raw parameter
        self.register_parameter(
            name='raw_chol_11', parameter=torch.nn.Parameter(chol_11)
        )
     # register the raw parameter
        self.register_parameter(
            name='raw_chol_22', parameter=torch.nn.Parameter(chol_22)
        )
        
        
        # register the raw parameter
        self.register_parameter(
            name='raw_chol_21', parameter=torch.nn.Parameter(chol_21)
        )
        
        


        # register the constraint
        self.register_constraint("raw_chol_11", pos_constraint)
                # register the constraint
        self.register_constraint("raw_chol_22", pos_constraint)
        
        @property
        def chol_11(self):
            return self.raw_chol_11_constraint.transform(self.raw_chol_11)
            
        @chol_11.setter
        def chol_11(self, value):
            return self._set_chol_11(value)
            
        
        def _set_chol_11(self, value):
            if not torch.is_tensor(value):
                value = torch.as_tensor(value).to(self.raw_chol_11)
            self.initialize(raw_chol_11 = self.raw_chol_11_constraint.inverse_transform(value))
            
            


        
        
        @property
        def chol_22(self):
            return self.raw_chol_22_constraint.transform(self.raw_chol_22)
            
        @chol_22.setter
        def chol_22(self, value):
            return self._set_chol_22(value)
            
        
        def _set_chol_22(self, value):
            if not torch.is_tensor(value):
                value = torch.as_tensor(value).to(self.raw_chol_22)
                
            self.initialize(raw_chol_22= self.raw_chol_22_constraint.inverse_transform(value))
        
        
        
        @property
        def chol_21(self):
            return (self.raw_chol_21)
            
        @chol_21.setter
        def chol_21(self, value):
            return self._set_chol_21(value)
            
        
        def _set_chol_21(self, value):
            if not torch.is_tensor(value):
                value = torch.as_tensor(value).to(self.raw_chol_21)
                
            self.initialize(raw_chol_21= (value))

        
        
       # self.chol_q = chol_q

#
#    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, nugget = False, **params):
#        k = self.data_covar_module
#       # chol_factor = self.chol_q
#
#        chol_factor = torch.zeros(self.num_tasks, self.num_tasks)
#        chol_factor[0,0] = self.raw_chol_11
#        chol_factor[1,1] = self.raw_chol_22
#        chol_factor[1,0] = self.raw_chol_21
#
#
#       # chol_factor = lazify(chol_factor)
#        kappa = gpytorch.matmul(chol_factor, chol_factor.t())
#
#        #x1 = x1.reshape(x1.shape[0], 1)
#        #x2 = x2.reshape(x2.shape[0], 1)
#        #y2 = x2
#
#        k_fwd = (k.forward(x1, x2))
#
#        kronecker_kernel = torch.kron(k_fwd, kappa)
#
#        if (kronecker_kernel.shape[0] == kronecker_kernel.shape[1]):
#            if (nugget == True):
#                kronecker_kernel = kronecker_kernel+ 1e-6 * torch.eye(kronecker_kernel.shape[0])
#
#        return kronecker_kernel
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

#
#
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("MultitaskKernel does not accept the last_dim_is_batch argument.")
        covar_i = self.task_covar_module.covar_matrix.evaluate()
        if len(x1.shape[:-2]):
            covar_i = covar_i.repeat(*x1.shape[:-2], 1, 1)
        covar_x = (self.data_covar_module.forward(x1, x2, **params))
        res = gpytorch.lazy.lazify(torch.kron(covar_x, covar_i))
        
        return res.diag() if diag else res
    
    
  
