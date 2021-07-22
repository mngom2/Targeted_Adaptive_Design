import math
import torch
from torch import nn
from torch import Tensor
import gpytorch
from copy import deepcopy

from torch.nn import ModuleList

from .mean import Mean


from functools import partial
from matplotlib import pyplot as plt



class TensorProductSubMean(Mean):
    """
    Class to get the tensorproduct mean
    """

    def __init__(self, agg_data, C, num_tasks, **kwargs):
        self.num_tasks = num_tasks
        self.agg_data = agg_data
        self.kermatrix = C

        
        
    def forward(self):
      
        C_ = self.kermatrix
        self.agg_data = x
        one_vec = torch.ones(C_.shape[1], 1)
        den = gpytorch.inv_quad(C_, one_vec)
        num = gpytorch.inv_matmul(C_, x, one_vec.t())
        
        return (num/den) * one_vec
        
        
        
        
    

        
        

    
    
  
