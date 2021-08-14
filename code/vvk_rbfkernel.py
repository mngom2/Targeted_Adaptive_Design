import math
import torch
from torch import nn
from torch import Tensor
import gpytorch
from gpytorch.kernels import RBFKernel
from gpytorch.constraints import Positive
from gpytorch.functions import RBFCovariance
from gpytorch.settings import trace_mode



def postprocess_rbf(dist_mat):
    return dist_mat.div_(-2).exp_()




class vvkRBFKernel(RBFKernel):
    """
    Class to get the tensorproduct kernel
    """

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):
            x1_ = x1.clone().div(self.lengthscale)
            
            x2_ = x2.clone().div(self.lengthscale)
            
            
            return self.covar_dist(
                x1_, x2_, square_dist=True, diag=diag, dist_postprocess_func=postprocess_rbf, postprocess=True, **params
            )
       
        return RBFCovariance.apply(
            x1,
            x2,
            self.lengthscale,
            lambda x1, x2: self.covar_dist(
                x1, x2, square_dist=True, diag=False, dist_postprocess_func=postprocess_rbf, postprocess=False, **params
            ),
        )

        
        

    
    
  
