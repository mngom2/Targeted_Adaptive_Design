import math
import torch
import gpytorch
from matplotlib import pyplot as plt







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
        
        

    
        
    def get_ell(f_target, x,theta2, theta1, g_theta, mu, K, cov_noise1, cov_noise2):
    
        g_theta1 = g_theta(theta1)
        g_theta2 = g_theta(theta2)
        
        
       
        
        
        Cff = K.forward(x,x)
        Cf1 = K.forward(x,g_theta1)
        Cf2 = K.forward(x, g_theta2)
        C11 = K(g_theta1, g_theta1.t()) + cov_noise1
        C12 = K(g_theta1, g_theta2.t())
        C22 = K(g_theta2, g_theta2.t()) + cov_noise2
        
        rhs_g1 = g_theta1 - mu.forward(g_theta1)

        pf1 = mu.forward(x) + C11.inv_matmul(rhs_g1, Cf1)
        
        C21 = C12.t()
        Q21 = C22 - C11.inv_quad(C12, C21)

        Cf1 = C1f.t()
        Qf1 = C22 - C11.inv_matmul(C1f, Cf1)
        
        
        
        second_term_Qf12 = C2f - C11.inv_matmul(C1f, C21)
        t_second_term_Qf12 = second_term_Qf12.t()

        Qf12_sec_term = Q21.inv_matmul(second_term_Qf12, t_second_term_Qf12)
        Qf12 = Qf1 - Qf12_sec_term

        
        inv_quad_Qf12, logdet_Qf12 = Qf12.inv_quad_logdet(f_target - pf1)
        trace_term_arg = Qf12.inv_matmul(right_tensor = None, left_tensor = Qf1)
        diag_trace_term_arg = trace_term_arg.diag()
        trace_term = diag_trace_term_arg .sum() - diag_trace_term_arg.size()
        
        ell = -1./2. * ( logdet_Qf12 + inv_quad_Qf12 + trace_term )
        
        
        
        return ell
        
        
    def get_ell(f_target, x,theta, g_theta, mu, K, cov_noise):
    
        g_theta = g_theta(theta)
        
        C_x = K.forward(x, x)
        C_theta = K.forward(g_theta, g_theta.t()) + cov_noise
        C_x_theta = K.forward(x, g_theta.t())
        rhs_g = g_theta - mu.forward(g_theta)
        mu_x = mu.forward(x)
        
        m = mu_x + C_theta.inv_matmul(rhs_g, C_x_theta)
        
        Q = C_x - C_x_theta.inv_quad(C_x_theta.t())
        
        inv_quad_Q, logdet_Q = Q.inv_quad_logdet(f_target - m)
        
        pll = -1/2. * (logdet_Q + inv_quad_Q)
        
        return pll
    

        
        
        
class OptimizationKernel(object):
    def get_multitask_mean(self):
        return 0
        
    def get_OK_covar_module(self):
        
        return 0
        
        
    def get_gaussian_likelihood(self):
        return 0
    
    
  
