from typing import Any

import math
import torch
from torch import nn
from torch import Tensor
import gpytorch
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.distributions import base_distributions
from gpytorch.settings import max_cholesky_size
from matplotlib import pyplot as plt






max_cholesky_size._set_value(2000)

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
        has_task_noise=False, _cov_noise1 = None, _cov_noise2 = None, rank=0, task_covar_prior=None, **kwargs):
        
        super().__init__(num_tasks, rank,
        task_prior,
        batch_shape,
        noise_prior,
        noise_constraint,
        has_global_noise,
        has_task_noise=False)

        self.cov_noise1 = _cov_noise1
        self.cov_noise2 = _cov_noise2
        

        
        
    def get_mll(self, out_data,  model_out):
        
        m0 = model_out.loc
        #print(m0)

        C11= model_out.covariance_matrix
       
       # print(C11)
        #print(model_out.lazy_covariance_matrix.batch_shape)
        covar = model_out.lazy_covariance_matrix
        diff = out_data - m0
        if diff.shape[:-1] != covar.batch_shape:
            print('blll')

        inv_quad_C11, logdet_C11 = gpytorch.inv_quad_logdet(C11, diff.unsqueeze(-1), logdet=True)
        pi = Tensor([math.pi])
        N = C11.shape[1]
        return 1./N * (-.5 * logdet_C11 - .5 * inv_quad_C11)
        

        
    def get_ell(self, agg_data, f_target, x, g_theta1, g_theta2, model, likelihood): #, cov_noise1, cov_noise2):

        """
        computes the expected ll

        """
       
       # print(max_cholesky_size.value())
        
        cov_noise1 = likelihood._shaped_noise_covar([g_theta1.shape[0],g_theta1.shape[0]]).evaluate()
        cov_noise2 = likelihood._shaped_noise_covar([g_theta2.shape[0],  g_theta2.shape[0]]).evaluate()

         
        mu = model.mean_module
        K = model.covar_module


        Cff = K.forward(x,x)
       
        Cf1 = K.forward(x,g_theta1)
        C11 = K.forward(g_theta1, g_theta1, nugget = False) + cov_noise1
     
        Cf2 = K.forward(x, g_theta2)
        
        C12 = K.forward(g_theta1, g_theta2)
        C22 = K.forward(g_theta2, g_theta2, nugget = False) + cov_noise2 #+ 0.005 * torch.eye(cov_noise2.shape[0])     #cov_noise2
    
        mean_data = mu.forward(g_theta1)
        
        mean_data = torch.flatten(mean_data)    #mean_data.reshape(mean_data.shape[0] * mean_data.shape[1], 1)
        #agg_data = agg_data.reshape(mean_data.shape[0], 1)
        noise1_vec = torch.sqrt(cov_noise1.diag())
        rhs_g1 =  agg_data  - mean_data #.reshape(mean_data.shape, 1) #g-S #+ noise1_vec #+ 0.005 * torch.ones(agg_data.shape)
        
        
        
        mean_x = mu.forward(x)
      
        mean_x = torch.flatten(mean_x)  #mean_x.reshape(mean_x.shape[0] * mean_x.shape[1], 1)
        
        pf1 = mean_x.reshape(mean_x.shape, 1) + C11.inv_matmul(rhs_g1, Cf1.evaluate())
         
        C21 = C12.t()
        Q21 = C22 - C11.inv_matmul(C12.evaluate(), C21.evaluate())

        
        C1f = Cf1.t()
        Qf1 = Cff - C11.inv_matmul(C1f.evaluate(), Cf1.evaluate()) #
        
       

        C2f = Cf2.t()
        second_term_Qf12 =  C2f - C11.inv_matmul(C1f.evaluate(), C21.evaluate())

        try:
            Qf12_sec_term = Q21.inv_matmul(second_term_Qf12.evaluate(), second_term_Qf12.t().evaluate())
        except:
            print('gtheta1')
            print(g_theta1)
            print('gtheta2')
            print(g_theta2)
            print('nugget')
            print(cov_noise1)
            print(C22)
            print(C11.inv_matmul(C12.evaluate(), C21.evaluate()))
            print(Q21)
            print(pf1)
            print(torch.det(C11))
            print(torch.det(K.forward(g_theta1, g_theta1, nugget = False)))
            for param_name, param in model.named_parameters():
                print(param_name)
                print(param)
            raise
        
        Qf12 = Qf1 - Qf12_sec_term
        
        pf1 = pf1.reshape(f_target.shape)
        
        
        try:
            inv_quad_Qf12 = Qf12.inv_matmul(f_target - pf1,  (f_target - pf1).t())
        except:
            print('gtheta1')
            print(g_theta1)
            print('gthe2')
            print(g_theta2)
            print('nugget')
            print(cov_noise1)
            print('Qf12')
            print(Qf12)
            print('Qf12_sec_term')
            print(Qf12_sec_term)
            print('Qf1')
            print(Qf1)
            print('Cff')
            print(Cff)
            print('invC11C1f')
            print(C11.inv_matmul(C1f.evaluate(), C21.evaluate()))
            print('C1f')
            print(C1f)
            print('Cf1')
            print(Cf1)
            print('C22')
            print(C22)
            print(Q21)
            print(pf1)
            print(torch.det(C11))
            print(torch.det(K.forward(g_theta1, g_theta1, nugget = False)))
            for param_name, param in model.named_parameters():
                print(param_name)
                print(param)
            
            raise
     
        logdet_Qf12 = Qf12.logdet()
        # simplified trace expression
        trace_term_arg = Qf12.matmul(Qf12_sec_term)
        #trace_term_arg = gpytorch.inv_matmul(Qf12,Qf1)
        diag_trace_term_arg = trace_term_arg.diag()

        trace_term = diag_trace_term_arg.sum()
        #N = Qf12.shape[0]

        ell = -1./2. * ( logdet_Qf12 + inv_quad_Qf12 + trace_term)
        lower_bound = torch.zeros(pf1.shape)
        upper_bound = torch.zeros(pf1.shape)
       
        for i in range(pf1.shape[0]):
            lower_bound[i] = pf1[i] -  torch.sqrt(Qf1[i,i])
            upper_bound[i] = pf1[i] +  torch.sqrt(Qf1[i,i])
            
       
        return ell, lower_bound, upper_bound
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
    

        
        

    
    
  
