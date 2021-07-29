import math
import torch
from torch import nn
from torch import Tensor
import gpytorch
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from matplotlib import pyplot as plt



class TensorProductLikelihood(MultitaskGaussianLikelihood):
    """
    Class to get the Likelihood
    """

    def __init__(self, x_new, num_tasks,
        task_prior=None,
        batch_shape=torch.Size(),
        noise_prior=None,
        noise_constraint=None,
        has_global_noise=True,
        has_task_noise=True, _cov_noise1 = None, _cov_noise2 = None, rank=1, task_covar_prior=None, **kwargs):
        super().__init__(num_tasks, rank,
        task_prior,
        batch_shape,
        noise_prior,
        noise_constraint,
        has_global_noise,
        has_task_noise=True)

        self.cov_noise1 = _cov_noise1
        self.cov_noise2 = _cov_noise2


    def get_hpll(self, out_data, g_theta1,mu, K):
#        g_theta1 = g_theta(theta1)
        
        C11 = K.forward(g_theta1, g_theta1.t())
        out_data = out_data.reshape(C11.shape[1], 1)
        m0 = mu.forward(g_theta1)
        m0 = m0.reshape(out_data.shape[0], 1)
#        one_vec = torch.ones(C11.shape[1], 1)
#        den = gpytorch.inv_quad(C11, one_vec)
#        num = gpytorch.inv_matmul(C11, out_data, one_vec.t())
#
#        m0 = (num/den) * one_vec
        
        
        inv_quad_C11 = gpytorch.inv_quad(C11, out_data - m0)
        logdet_C11 = gpytorch.logdet(C11)
        return -.5 * logdet_C11 - .5 * inv_quad_C11
        

        
    def get_ell(self, agg_data, f_target, x, g_theta1, g_theta2, mu, K): #, cov_noise1, cov_noise2):

        """
        computes the expected ll

        """

#        g_theta1 = g_theta(theta1)
#        g_theta2 = g_theta(theta2)
        cov_noise1 = self.cov_noise1
        cov_noise2 = self.cov_noise2




        Cff = K.forward(x,x)
        Cf1 = K.forward(x,g_theta1.t())
        Cf2 = K.forward(x, g_theta2.t())
        C11 = K.forward(g_theta1, g_theta1.t()) #+ cov_noise1

        C12 = K.forward(g_theta1, g_theta2.t())
        C22 = K.forward(g_theta2, g_theta2.t()) #+ cov_noise2

        mean_data = mu.forward(agg_data)
        mean_data = mean_data.reshape(mean_data.shape[0] * mean_data.shape[1], 1)
        agg_data = agg_data.reshape(mean_data.shape[0], 1)
        rhs_g1 =  agg_data- mean_data #g-S
      
        
        
        mean_x = mu.forward(x)
        mean_x = mean_x.reshape(mean_x.shape[0] * mean_x.shape[1], 1)

        pf1 = mean_x + gpytorch.inv_matmul(C11, rhs_g1, Cf1)

        C21 = C12.t()
        print(C22.shape)
        Q21 = C22 - gpytorch.inv_quad(C11, C12)

        C1f = Cf1.t()
        Qf1 = Cff - gpytorch.inv_quad(C11, C1f) #


        C2f = Cf2.t()
        second_term_Qf12 = C2f - gpytorch.inv_matmul(C11,C1f, C21)
      #  t_second_term_Qf12 = second_term_Qf12.t()
    

        Qf12_sec_term = gpytorch.inv_quad(Q21, second_term_Qf12)
        Qf12 = Qf1 - Qf12_sec_term


        inv_quad_Qf12 = gpytorch.inv_quad(Qf12, f_target - pf1)
        logdet_Qf12 = gpytorch.logdet(Qf12)
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
    

        
        

    
    
  
