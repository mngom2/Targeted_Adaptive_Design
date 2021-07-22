import math
import torch
from torch import nn
from torch import Tensor
import gpytorch
from matplotlib import pyplot as plt



class TensorProductLikelihood():
    """
    Class to get the Likelihood
    """

    def __init__(self, data_covar_module, _agg_data, x_new, num_tasks, _cov_noise1, _cov_noise2, rank=1, task_covar_prior=None, **kwargs):
        self.num_tasks = num_tasks
        self.rank = rank
        self.data_covar_module = data_covar_module
        self.task_covar_prior = task_covar_prior
        self.agg_data = _agg_data
        self.cov_noise1 = _cov_noise1
        self.cov_noise2 = _cov_noise2
        # register the raw parameter
        self.register_parameter(
            name='new_setting', parameter=torch.nn.Parameter(x_new)
        )

        
        

        
    def get_ell(self,f_target, x, theta1, theta2, g, g_theta, mu, K): #, cov_noise1, cov_noise2):

        """
        computes the expected ll

        """
#        theta2 = self.theta2
        g_theta1 = g_theta(theta1)
        g_theta2 = g_theta(theta2)
        agg_data = self.agg_data
        cov_noise1 = self.cov_noise1
        cov_noise2 = self.cov_noise2




        Cff = K(x,x)
        Cf1 = K(x,g_theta1.t())
        Cf2 = K(x, g_theta2.t())
        C11 = K(g_theta1, g_theta1.t()) + cov_noise1

        C12 = K(g_theta1, g_theta2.t())
        C22 = K(g_theta2, g_theta2.t()) + cov_noise2

        mean_data = mu(C11, agg_data)
        rhs_g1 = g - mean_data
        C21 = C12.t()

        Q21 = C22 - gpytorch.inv_quad(C11, C12)

        pf1 = mean_data + gpytorch.inv_matmul(C11, rhs_g1, Cf1)

        C21 = C12.t()
        Q21 = C22 - gpytorch.inv_quad(C11, C12)

        C1f = Cf1.t()
        Qf1 = C22 - gpytorch.inv_matmul(C11, C1f, Cf1)


        C2f = Cf2.t()
        second_term_Qf12 = C2f - gpytorch.inv_matmul(C11,C1f, C21)
        t_second_term_Qf12 = second_term_Qf12.t()


        Qf12_sec_term = gpytorch.inv_matmul(Q21, second_term_Qf12, t_second_term_Qf12)
        Qf12 = Qf1 - Qf12_sec_term


        inv_quad_Qf12, logdet_Qf12 = gpytorch.inv_quad_logdet(Qf12, f_target - pf1)
        # simplified trace expression
        trace_term_arg = gpytorch.inv_matmul(Qf12,Qf1)
        diag_trace_term_arg = trace_term_arg.diag()
        trace_term = diag_trace_term_arg .sum()



        ell = -1./2. * ( logdet_Qf12 + inv_quad_Qf12 + trace_term )



        return ell
#
#
#
#    def get_pll(f_target, x,theta, g_theta, mu, K, cov_noise):
#
#        """
#        computes the predicted ll needed for the first iteration of the algorithm
#
#        """
#
#        g_theta = g_theta(theta)
#
#        C_x = K.forward(x, x)
#        C_theta = K.forward(g_theta, g_theta.t()) + cov_noise
#        C_x_theta = K.forward(x, g_theta.t())
#        rhs_g = g_theta - mu.forward(g_theta)
#        mu_x = mu.forward(x)
#
#        m = mu_x + C_theta.inv_matmul(rhs_g, C_x_theta)
#
#        Q = C_x - C_x_theta.inv_quad(C_x_theta.t())
#
#        inv_quad_Q, logdet_Q = Q.inv_quad_logdet(f_target - m)
#
#        pll = -1/2. * (logdet_Q + inv_quad_Q)
#
#        return pll
    

        
        

    
    
  
