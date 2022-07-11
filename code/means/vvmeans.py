import math
import torch
from torch import nn
from torch import Tensor
import gpytorch
from copy import deepcopy

from torch.nn import ModuleList

from gpytorch.means import MultitaskMean


from functools import partial
from matplotlib import pyplot as plt

"""
custom means
"""

class TensorProductSubMean(MultitaskMean):
    """
    Class to get the tensorproduct mean
    """

    def __init__(self, base_means, num_tasks, **kwargs):
        super().__init__(base_means, num_tasks)
       # self.kermatrix = _C
        
        
#    def forward(self, input):
##
#        C = vvk.
#        one_vec = torch.ones(C.shape[1], 1)
#        den = gpytorch.inv_quad(C, one_vec)
#        num = gpytorch.inv_matmul(C, input, one_vec.t())
#
#        return (num/den) * one_vec
#
        
        
        
    

        
        

    
    
  
