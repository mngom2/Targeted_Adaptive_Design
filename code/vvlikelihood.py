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

    def __init__(self, num_tasks,
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



        
        
    def get_mll(self, out_data,  model_out):
    
        m0 = model_out.loc
 

        C11= model_out.covariance_matrix
        

        
        diff = out_data - m0
       

        inv_quad_C11, logdet_C11 = gpytorch.inv_quad_logdet(C11, diff.unsqueeze(-1), logdet=True)
        pi = Tensor([math.pi])
        N = C11.shape[1]
        return (-.5 * logdet_C11 - .5 * inv_quad_C11 - N/2 * torch.log(2 * pi))
        

        
    def get_ell(self, agg_data, f_target, x, g_theta1, g_theta2, mu, K): #, cov_noise1, cov_noise2):

        """
        computes the expected ll

        """

#       g_theta1 = g_theta(theta1)
#       g_theta2 = g_theta(theta2)
        cov_noise1 = self.cov_noise1
        cov_noise2 = self.cov_noise2




        Cff = K.forward(x,x)
        
        Cf1 = K.forward(x,g_theta1)
        Cf2 = K.forward(x, g_theta2)
        C11 = K.forward(g_theta1, g_theta1) #+ cov_noise1
        C12 = K.forward(g_theta1, g_theta2)
        C22 = K.forward(g_theta2, g_theta2) #+ cov_noise2
        
        mean_data = mu.forward(g_theta1)
        mean_data = torch.flatten(mean_data)    #mean_data.reshape(mean_data.shape[0] * mean_data.shape[1], 1)
        #agg_data = agg_data.reshape(mean_data.shape[0], 1)
        rhs_g1 =  agg_data- mean_data.reshape(mean_data.shape, 1) #g-S
        
        
        
        mean_x = mu.forward(x)
        mean_x = torch.flatten(mean_x)  #mean_x.reshape(mean_x.shape[0] * mean_x.shape[1], 1)

        pf1 = mean_x.reshape(mean_x.shape, 1) + gpytorch.inv_matmul(C11, rhs_g1, Cf1)

        C21 = C12.t()
        Q21 = C22 - gpytorch.inv_matmul(C11, C12, C12.t())

    
        
        C1f = Cf1.t()
        Qf1 = Cff - gpytorch.inv_matmul(C11, C1f, C1f.t()) #
       

        C2f = Cf2.t()
        second_term_Qf12 = C2f - gpytorch.inv_matmul(C11,C1f, C21)

        

        Qf12_sec_term = gpytorch.inv_matmul(Q21, second_term_Qf12, second_term_Qf12.t())
        Qf12 = Qf1 - Qf12_sec_term
        
        
        inv_quad_Qf12 = gpytorch.inv_quad(Qf12,  f_target - pf1)
     
        logdet_Qf12 = gpytorch.logdet(Qf12)
        # simplified trace expression
        trace_term_arg = gpytorch.inv_matmul(Qf12,Qf1)
        diag_trace_term_arg = trace_term_arg.diag()
        trace_term = diag_trace_term_arg.sum()


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
    

        
        

    
    
  
