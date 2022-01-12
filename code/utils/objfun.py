# Support for simulating a vector field over a hypercube, with a known value at
# a selected point.
#
# Carlo Graziani, ANL
#
"""
Support for simulating a vector field over a hypercube, with a known value at a
selected point.
"""

import numpy as np
from scipy.stats import uniform
import torch
from torch import Tensor


######################################################################
######################################################################
######################################################################
class ObjFun(object):
    """
    An N-dimensional vector field over a D-dimensional hypercube, with
    a given value at a given point.
    """
#########################################
    def __init__(self, D=2, N = 2, low = -3., high = 3.,  tgt_loc=Tensor(np.array([[0.8731,0.5664]])),
                 tgt_vec = Tensor(np.array([3.0173/8.9280,3.1267/8.9280]))):

        self.tgt_loc = tgt_loc
        self.tgt_vec= tgt_vec
        self.D = D
        self.N = N
        self.low = low
        self.high = high
    def _get_obj(self, x1_, x2_):

        J1 = 3.* (1. - x1_) ** 2. * torch.exp(- x1_ **2 - (x2_ + 1) **2 ) - 10. * (x1_/5. - x1_ **3 - x2_**5)*torch.exp(-x1_**2 - x2_ **2) - 3 * torch.exp(- x2_ **2 - (x1_ + 2) **2 ) + 0.5 * (2 * x1_ + x2_)
        
        
        
        J2 = 3. * (1. + x2_) ** 2. * torch.exp(-x2_ **2 - (-x1_ + 1) **2 ) - 10. * (-x2_/5. + x2_ **3 + x1_**5)*torch.exp(-x2_**2 - x1_ **2) - 3. * torch.exp(- x1_ **2 - (-x2_ + 2) **2 ) #+ 0.5 * (-2 * x2_+x1_)
        J1 = J1.reshape(J1.shape[0], 1)
        J2 = J2.reshape(J2.shape[0], 1)
        
        Vf_1 = J1/8.9280
        Vf_2 = J2/8.9280
        
        return torch.cat((Vf_1, Vf_2),1)
    
    def __call__(self, x1_, x2_):
        return self._get_obj(x1_, x2_)
      
           
