import math
import torch
import gpytorch
from matplotlib import pyplot as plt





def cholesky_tensor_product(A, B):
    return 0 

class TensorProductKernel(object):
    """
    Class to get the tensorproduct kernel
    """

    def __init__(self, base_means, data_covar_module, num_tasks, rank=1, task_covar_prior=None, **kwargs):
        self.num_tasks = num_tasks
        self.rank = rank
        self.base_means = base_means
        self.data_covar_module = data_covar_module
        self.task_covar_prior = task_covar_prior
        
        
    def get_TPK_mean(self):
    
        return gpytorch.means.MultitaskMean(self.base_means, self.num_tasks)
        
        
        
    def get_TPK_covar_module(self):
        
        return gpytorch.kernels.MultitaskKernel(
            self.data_covar_module, self.num_tasks, self.rank)
        
        
    def get_gaussian_likelihood(self):
        return gpytorch.likelihoods.MultitaskGaussianLikelihood(self.num_tasks)
    
    def get_pll(f_target, x,theta_2, theta_1, g):
        mu = self.get_TPK_mean()
        K = self.get_TPK_covar_module()()
        
        C_ff = K(x)
        C_f1 = 0
        C_f2 = 0
        C_11 = 0
        C_12 = 0
        C_22 = 0
        p_f1 = 0
        Q_21 = 0
        Q_f1 = 0
        Q_f12 = 0
        
        
        m = mu(x)
        
        return 0
        
        
        
class OptimizationKernel(object):
    def get_multitask_mean(self):
        return 0
        
    def get_OK_covar_module(self):
        
        return 0
        
        
    def get_gaussian_likelihood(self):
        return 0
    
    
  
