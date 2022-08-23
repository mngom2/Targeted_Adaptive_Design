from typing import Any, Optional
import warnings
from copy import deepcopy
import math
import torch
from torch import nn
from torch import Tensor
import gpytorch
from gpytorch.likelihoods import MultitaskGaussianLikelihood, _MultitaskGaussianLikelihoodBase, _GaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import MultitaskHomoskedasticNoise, FixedGaussianNoise, Noise
from .noise_models_multitask_fixed_gaussian import MultitaskFixedGaussianNoise
from .likelihoods_multitask_fixed_gaussian import MultitaskFixedNoiseGaussianLikelihood
from gpytorch.distributions import base_distributions
from gpytorch.settings import max_cholesky_size
from gpytorch.lazy import ZeroLazyTensor, DiagLazyTensor
from gpytorch.utils.warnings import GPInputWarning
from matplotlib import pyplot as plt





#torch.set_default_dtype(torch.float64)


#gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False)
#with torch.no_grad():
gpytorch.settings.fast_computations(covar_root_decomposition = True, log_prob = True, solves = False)
gpytorch.settings.max_cg_iterations(100)
gpytorch.settings.max_preconditioner_size(100)
max_cholesky_size._set_value(3000)
#gpytorch.settings.fast_pred_var(False)

use_cuda = torch.cuda.is_available()

def barrierFunction(x, low, high, c):
    out = 0.
    n = x.shape[0]
    zero_tensor = Tensor([0.]).to(x.device)
    for i in range(n):
        out = out + c * ( (torch.max(zero_tensor, (-x[i,0] - 3.))) ** 2. + (torch.max(zero_tensor, (x[i,0] - 3.))) ** 2. + (torch.max(zero_tensor, (-x[i,1] - 3.))) ** 2. + (torch.max(zero_tensor, (x[i,1] - 3.))) ** 2.)
    return out #c * ( (torch.max(Tensor([0.]), (-x[0,0] - 3.))) ** 2. + (torch.max(Tensor([0.]), (x[0,0] - 3.))) ** 2. + (torch.max(Tensor([0.]), (-x[0,1] - 3.))) ** 2. + (torch.max(Tensor([0.]), (x[0,1] - 3.))) ** 2.)

class TensorProductLikelihood(MultitaskGaussianLikelihood):
    """
    Class to get the different loss function for fitted noise (i.e not fixed noise)
    """
    def __init__(self, num_tasks,
        task_prior=None,
        batch_shape=torch.Size(),
        noise_prior=None,
        noise_constraint=None,
        has_global_noise=True,
        has_task_noise=False, _cov_noise1 = None, _cov_noise2 = None, rank=0, **kwargs):
        
        super().__init__(num_tasks,
        rank)

        self.cov_noise1 = _cov_noise1
        self.cov_noise2 = _cov_noise2
        

        
    def get_mll(self, agg_data, g_theta1,  model, likelihood, noise_value):
        
        output = model(g_theta1)
        output_ll = likelihood(output)
        m0 = output_ll.loc
        C11= output_ll.covariance_matrix
       
        covar = output_ll.lazy_covariance_matrix
        diff = agg_data - m0
        if diff.shape[:-1] != covar.batch_shape:
            print('error shape')
        inv_quad_C11, logdet_C11 = gpytorch.inv_quad_logdet(C11, diff.unsqueeze(-1), logdet=True)
        pi = Tensor([math.pi])
        N = C11.shape[1]
        return 1./N * (-.5  * logdet_C11 - .5  * inv_quad_C11), inv_quad_C11
        

        
    def get_ell(self, agg_data, f_target, x, g_theta1, model, likelihood,noise_value, g_theta2): #, cov_noise1, cov_noise2):

        """
        computes the expected ll

        """
        
        #with gpytorch.settings.fast_pred_var():
       
        cov_noise1 = noise_value * torch.eye(agg_data.shape[0])   #likelihood._shaped_noise_covar([g_theta1.shape[0],g_theta1.shape[0]]).evaluate()
        cov_noise2 = noise_value * torch.eye(2* g_theta2.shape[0])#likelihood._shaped_noise_covar([g_theta2.shape[0],  g_theta2.shape[0]]).evaluate()
#        print(cov_noise1)
#        print(cov_noise2)
        #cov_noisex = likelihood._shaped_noise_covar([x.shape[0],  x.shape[0]]).evaluate()
         
        mu = model.mean_module
        K = model.covar_module


        Cff = K.forward(x,x) #+ cov_noisex
       
        Cf1 = K.forward(x,g_theta1)
        
        C11 = K.forward(g_theta1, g_theta1) + cov_noise1
  
     
        Cf2 = K.forward(x, g_theta2)
        
        C12 = K.forward(g_theta1, g_theta2)
        C22 = K.forward(g_theta2, g_theta2) + cov_noise2
    
        mean_data = mu.forward(g_theta1)
        
        mean_data = torch.flatten(mean_data)
        rhs_g1 =  agg_data  - mean_data
        
        
        
        mean_x = mu.forward(x)
      
        mean_x = torch.flatten(mean_x)
        
        pf1 = mean_x.reshape(mean_x.shape, 1) + C11.inv_matmul(rhs_g1, Cf1.evaluate())
         
        C21 = C12.t()
        Q21 = C22 - C11.inv_matmul(C12.evaluate(), C21.evaluate())
      

        
        C1f = Cf1.t()
        Qf1 = (Cff - C11.inv_matmul(C1f.evaluate(), Cf1.evaluate())) #
        
    

        C2f = Cf2.t()
        second_term_Qf12 =  C2f - C11.inv_matmul(C1f.evaluate(), C21.evaluate())
        
        try:
            Qf12_sec_term = Q21.inv_matmul(second_term_Qf12.evaluate(), second_term_Qf12.t().evaluate())
        except:
            print('add/increase jitter for Q21')
            raise
        
        Qf12 = Qf1 - Qf12_sec_term
        #Qf12 = Qf12.add_jitter(1e-8)
    
        pf1 = pf1.reshape(f_target.shape)
        
        
        try:
            inv_quad_Qf12 = Qf12.inv_matmul(f_target - pf1,  (f_target - pf1).t())
        except:
            print('add/increase jitter for Qf12')
            
            raise
     
        logdet_Qf12 = Qf12.logdet()
        # simplified trace expression
        trace_term_arg = gpytorch.inv_matmul(Qf12,Qf12_sec_term) #Qf12.matmul(Qf12_sec_term) #
        #trace_term_arg = gpytorch.inv_matmul(Qf12,Qf1)
        diag_trace_term_arg = trace_term_arg.diag()

        trace_term = diag_trace_term_arg.sum()
      
        N = C11.shape[0]

        ell = -1./2. * (  logdet_Qf12 +  inv_quad_Qf12 +   trace_term ) #- (Qf1[0,0]**2 + Qf1[0,0]**2)
        ell = ell -  barrierFunction(x, -3., 3., 1000.) #-  barrierFunction(g_theta2, -3., 3., 0.01)
        lower_bound = torch.zeros(pf1.shape)
        upper_bound = torch.zeros(pf1.shape)
       
        for i in range(pf1.shape[0]):
            lower_bound[i] = pf1[i] -  torch.sqrt(Qf1[i,i])
            upper_bound[i] = pf1[i] + torch.sqrt(Qf1[i,i])
        
        return ell, pf1, Qf1, Qf12, logdet_Qf12 +  inv_quad_Qf12
    
    """
    Computes the inverse quadratic formula (fdata - mean)^T mat^{-1}(fdata - mean)
    """
    def get_inv_quad(self, mat, fdata, g_theta, data_12, x, model, noise_value):

        cov_noise =  noise_value * torch.eye(data_12.shape[0])    #likelihood._shaped_noise_covar([ agg_data.shape[0],
        #cov_noise2 =  noise_value * torch.eye(2 * g_theta2.shape[0])
        K = model.covar_module
        mu = model.mean_module
        Cf1 = K.forward(x, g_theta)
        
        C11 = K.forward(g_theta, g_theta, add_jitter = True) + cov_noise
        
        rhs_g = data_12 - torch.flatten(mu.forward(g_theta))
        rhs_g = rhs_g.reshape(rhs_g.shape, 1)
        
        mean_x = mu.forward(x)
      
        mean_x = torch.flatten(mean_x)
       
        pf12 = mean_x.reshape(mean_x.shape, 1) + C11.inv_matmul(rhs_g, Cf1.evaluate())
    
  
        
        pf12 = pf12.reshape(fdata.shape)
        
        diff = fdata - pf12
        #print(diff.shape)
        #diff = diff.reshape(diff.shape[0], 1)
        inv_quad, logdet_C11 = gpytorch.inv_quad_logdet(mat.evaluate(), diff, logdet=True)
        return inv_quad





    def get_pll(self, f_target, x,g_theta, agg_data, model, likelihood):

        """
        computes the predictive ll

        """

        mu = model.mean_module
        K = model.covar_module

        Cff = K.forward(x, x)
        
        
        #cov_noise = likelihood._shaped_noise_covar([g_theta.shape[0],g_theta.shape[0]]).evaluate()
        
        C11 = K.forward(g_theta, g_theta) #+ cov_noise
        Cf1 = K.forward(x, g_theta)
        rhs_g = agg_data - (mu.forward(g_theta)).flatten()
        mu_x = mu.forward(x)
        
        
        mu_x = mu_x.flatten()
        C1f = Cf1.t()
        m = mu_x + C11.inv_matmul(rhs_g, Cf1.evaluate())


        Q = Cff - C11.inv_matmul(C1f.evaluate(), Cf1.evaluate())
        m = m.reshape(f_target.shape)
        inv_quad_Q, logdet_Q = Q.inv_quad_logdet(f_target - m, logdet = True)

        pll = 1./2. * (logdet_Q + inv_quad_Q)
        #pll = torch.linalg.norm((f_target - m), ord = 2)  #
        lower_bound = torch.zeros(m.shape)
        upper_bound = torch.zeros(m.shape)
       
        for i in range(m.shape[0]):
            lower_bound[i] = m[i] -  torch.sqrt(Q[i,i])
            upper_bound[i] = m[i] +  torch.sqrt(Q[i,i])

        return pll, lower_bound, upper_bound
        
        
"""
Onjective functions for the Multitask fixed noise likelihood case
""" 
class FixedNoiseMultitaskGaussianLikelihood(MultitaskFixedNoiseGaussianLikelihood):


    def __init__(self, noises, **kwargs):
        
        super().__init__(noises)
        
    def get_ell(self, agg_data, f_target,x_, g_theta1, model, likelihood, g_theta2, cov_noise1, cov_noise2):
    #def get_ell(self, agg_data, f_target, g_theta1, model, likelihood, noise_value, samples): #, cov_noise1, cov_noise2):

        """
        computes the expected ll (TAD optimization function)

        """
       
        #cov_noisex = noise_value * torch.eye(2* x.shape[0])
        #with gpytorch.settings.fast_pred_var():
#
#        x_ = samples[0:g_theta1.shape[1]]
#        try:
#            x_ = Tensor(x_.reshape(math.ceil(x_.shape[0]/g_theta1.shape[1]), g_theta1.shape[1]))
#        except:
#            print(samples.shape)
#            print(x_.shape)
#            print(g_theta1.shape[1])
#            print(samples[0])
#            print(samples[0][0:g_theta1.shape[1]])
#            raise
#        g_theta2 = samples[g_theta1.shape[1]:]
#        g_theta2 = Tensor(g_theta2.reshape(math.ceil(g_theta2.shape[0]/g_theta1.shape[1]), g_theta1.shape[1]))

        # Move these two to jupyter notebook Conv_Success
#         cov_noise1 =  noise_value * torch.eye(agg_data.shape[0])
#         cov_noise2 =  noise_value * torch.eye(2 * g_theta2.shape[0])
#
        mu = model.mean_module
        K = model.covar_module
        if use_cuda:
            mu = mu.cuda()
            K = K.cuda()

        Cff = K.forward(x_,x_, add_jitter = True)  #+ cov_noise
        Cf1 = K.forward(x_,g_theta1)
        C11 = K.forward(g_theta1, g_theta1, add_jitter = True) + cov_noise1
        
        
        
        Cf2 = K.forward(x_, g_theta2)
        
        C12 = K.forward(g_theta1, g_theta2)
        
        C22 = K.forward(g_theta2, g_theta2, add_jitter = True) + cov_noise2
        mean_data = mu.forward(g_theta1)
        #mean_data = mean_data.reshape(agg_data.shape)
        mean_data = torch.flatten(mean_data)    #mean_data.reshape(mean_data.shape[0] * mean_data.shape[1], 1)
        #agg_data = agg_data.reshape(mean_data.shape[0], 1)
       # noise1_vec = torch.sqrt(cov_noise1.diag())
        rhs_g1 =  agg_data  - mean_data #.reshape(mean_data.shape, 1) #g-S #+ noise1_vec #+ 0.005 * torch.ones(agg_data.shape)
        
        rhs_g1 = rhs_g1.reshape(rhs_g1.shape, 1)
        
        mean_x = mu.forward(x_)
      
        mean_x = torch.flatten(mean_x)  #mean_x.reshape(mean_x.shape[0] * mean_x.shape[1], 1)
        
        
        #with gpytorch.settings.cholesky_jitter(self.custom_jitter):
        pf1 = mean_x.reshape(mean_x.shape, 1) + C11.inv_matmul(rhs_g1, Cf1.evaluate())
         
        C21 = C12.t()
        #with gpytorch.settings.cholesky_jitter(self.custom_jitter):
        Q21 = C22 - C11.inv_matmul(C12.evaluate(), C21.evaluate())
        
#        print('here q21 ell')
#
       # Q21 =  Q21.add_jitter(1e-8)
       # print( torch.symeig(Q21.evaluate()) )
        
        #
        
        C1f = Cf1.t()
        
        #with gpytorch.settings.cholesky_jitter(self.custom_jitter):
        Qf1 = Cff - C11.inv_matmul(C1f.evaluate(), Cf1.evaluate()) #
        
       

        C2f = Cf2.t()
        #with gpytorch.settings.cholesky_jitter(self.custom_jitter):
        second_term_Qf12 =  C2f - C11.inv_matmul(C1f.evaluate(), C21.evaluate())

        try:
            #with gpytorch.settings.cholesky_jitter(self.custom_jitter):
            Qf12_sec_term = (Q21).inv_matmul(second_term_Qf12.evaluate(), second_term_Qf12.t().evaluate())
        except:

            raise
        
        Qf12 = Qf1 - Qf12_sec_term
       # Qf12 =  Qf12.add_jitter(1e-8)
        #print( torch.symeig(Qf12.evaluate()) )
        logdet_Qf12 = Qf12.logdet()
        
       
        pf1 = pf1.reshape(f_target.shape)
       
        
        
        try:
            #with gpytorch.settings.cholesky_jitter(self.custom_jitter):
            inv_quad_Qf12 = (Qf12).inv_matmul(f_target - pf1,  (f_target - pf1).t())
        except:

            raise

        
        # simplified trace expression
        #trace_term_arg = gpytorch.inv_matmul(Qf12,Qf12_sec_term) #Qf12.matmul(Qf12_sec_term) #
        trace_term_arg = gpytorch.inv_matmul(Qf12,Qf1.evaluate())
        diag_trace_term_arg = trace_term_arg.diag()

        trace_term = diag_trace_term_arg.sum()
        #N = Qf12.shape[0]
        
#        print('T')
#        print(Qf1.logdet())
#        print(logdet_Qf12)
#        print(inv_quad_Qf12)
#        print(trace_term)
#
##        print( 0.5 * torch.log(torch.det(Cff.evaluate()) / torch.det(Qf1.evaluate())))
##        print( 0.5 * torch.log(torch.det(Cff.evaluate()) / torch.det(Qf12.evaluate())) )
##        print( 0.5 * torch.log(torch.det(Qf1.evaluate()) / torch.det(Qf12.evaluate())) )
##        print(Qf12_sec_term)
##        print(torch.det(Qf12_sec_term))
#        print('end T')

        ell = -1./2. * (  logdet_Qf12 + inv_quad_Qf12 +  trace_term)
    
        #print(barrierFunction(x_, -3., 3., 1000.)  -  barrierFunction(g_theta2, -3., 3., 1000.))
        ell = ell - barrierFunction(x_, -1., 1., 10000.)  -  barrierFunction(g_theta2, -1., 1., 10000.)
        
        
#        print(pf1 - f_target)
        return ell, pf1, Qf1, Qf12, logdet_Qf12 +  inv_quad_Qf12, Q21

    def get_pll(self, f_target, x,g_theta, agg_data, model, likelihood,  noise_value):

        """
        computes the predictive ll

        """

        mu = model.mean_module
        K = model.covar_module

        Cff = K.forward(x, x)
        cov_noise1 =  noise_value * torch.eye(agg_data.shape[0])
        
        #cov_noise = likelihood._shaped_noise_covar([g_theta.shape[0],g_theta.shape[0]]).evaluate()
        
        C11 = K.forward(g_theta, g_theta) + cov_noise1
        Cf1 = K.forward(x, g_theta)
        rhs_g = agg_data - (mu.forward(g_theta)).flatten()
        mu_x = mu.forward(x)
        
        
        mu_x = mu_x.flatten()
        C1f = Cf1.t()
        #with gpytorch.settings.cholesky_jitter(self.custom_jitter):
        m = mu_x + C11.inv_matmul(rhs_g, Cf1.evaluate())

        #with gpytorch.settings.cholesky_jitter(self.custom_jitter):
        Q = Cff - C11.inv_matmul(C1f.evaluate(), Cf1.evaluate())
        m = m.reshape(f_target.shape)
        inv_quad_Q, logdet_Q = Q.inv_quad_logdet(f_target - m, logdet = True)

        pll = 1./2. * (logdet_Q + inv_quad_Q)
        pll = pll - barrierFunction(x, -3., 3., 100000.)
        #pll = torch.linalg.norm((f_target - m), ord = 2)  #
        lower_bound = torch.zeros(m.shape)
        upper_bound = torch.zeros(m.shape)
       
        for i in range(m.shape[0]):
            lower_bound[i] = m[i] -  torch.sqrt(Q[i,i])
            upper_bound[i] = m[i] +  torch.sqrt(Q[i,i])

        return pll, lower_bound, upper_bound
        
        
    def get_l2(self, f_target, x,g_theta, agg_data, model, likelihood, noise_value):

        """
        computes the predicted ll needed for the first iteration of the algorithm

        """

        mu = model.mean_module
        K = model.covar_module

        Cff = K.forward(x, x)
        cov_noise1 =  noise_value * torch.eye(agg_data.shape[0])
        
        #cov_noise = likelihood._shaped_noise_covar([g_theta.shape[0],g_theta.shape[0]]).evaluate()
        
        C11 = K.forward(g_theta, g_theta) + cov_noise1
        Cf1 = K.forward(x, g_theta)
        rhs_g = agg_data - (mu.forward(g_theta)).flatten()
        mu_x = mu.forward(x)
        
        
        mu_x = mu_x.flatten()
        C1f = Cf1.t()
        #with gpytorch.settings.cholesky_jitter(self.custom_jitter):
        m = mu_x + C11.inv_matmul(rhs_g, Cf1.evaluate())

        #with gpytorch.settings.cholesky_jitter(self.custom_jitter):
        Q = Cff - C11.inv_matmul(C1f.evaluate(), Cf1.evaluate())
        m = m.reshape(f_target.shape)
        inv_quad_Q, logdet_Q = Q.inv_quad_logdet(f_target - m, logdet = True)

        pll = 1./2. * (logdet_Q + inv_quad_Q)
        #pll = pll - barrierFunction(x, -3., 3., 100000.)
        pll = torch.linalg.norm((f_target - m), ord = 2)  #
        lower_bound = torch.zeros(m.shape)
        upper_bound = torch.zeros(m.shape)
       
        for i in range(m.shape[0]):
            lower_bound[i] = m[i] -  torch.sqrt(Q[i,i])
            upper_bound[i] = m[i] +  torch.sqrt(Q[i,i])

        return pll, lower_bound, upper_bound
        
        
    def get_mll(self, agg_data, g_theta1,  model, likelihood, noise_value):
        
        cov_noise1 =  noise_value * torch.eye(agg_data.shape[0])    #likelihood._shaped_noise_covar([ agg_data.shape[0],

        output = model(g_theta1)
        #output_ll = likelihood(output)
        m0 = model.mean_module.forward(g_theta1) #output_ll.loc
        #print(m0)
        #C11= output_ll.covariance_matrix + cov_noise1
        K = model.covar_module
 #       mu = model.mean_module.forward(g_theta1)
 
        C11= K.forward(g_theta1, g_theta1, add_jitter = True) + cov_noise1

        #covar = output_ll.lazy_covariance_matrix
        m0 = m0.reshape(agg_data.shape)
        diff = agg_data - m0
#        if diff.shape[:-1] != covar.batch_shape:
#            print('error shape')
    
        inv_quad_C11, logdet_C11 = gpytorch.inv_quad_logdet(C11, diff.unsqueeze(-1), logdet=True)
        pi = Tensor([math.pi])
        N = C11.shape[1]
        return 1./N * (-.5 * logdet_C11 - .5 * inv_quad_C11 ), inv_quad_C11
        
    def get_inv_quad(self, mat, fdata, g_theta, data_12, x, model, noise_value):

        cov_noise =  noise_value * torch.eye(data_12.shape[0])    #likelihood._shaped_noise_covar([ agg_data.shape[0],
        #cov_noise2 =  noise_value * torch.eye(2 * g_theta2.shape[0])
        K = model.covar_module
        mu = model.mean_module
        Cf1 = K.forward(x, g_theta)
        
        C11 = K.forward(g_theta, g_theta, add_jitter = True) + cov_noise
        
        rhs_g = data_12 - torch.flatten(mu.forward(g_theta))
        rhs_g = rhs_g.reshape(rhs_g.shape, 1)
        
        mean_x = mu.forward(x)
      
        mean_x = torch.flatten(mean_x)
       
        pf12 = mean_x.reshape(mean_x.shape, 1) + C11.inv_matmul(rhs_g, Cf1.evaluate())
    
  
        
        pf12 = pf12.reshape(fdata.shape)
        
        
        diff = fdata - pf12
#        print(diff.shape)
#        diff = diff.reshape(diff.shape[0], 1)
        inv_quad, logdet_C11 = gpytorch.inv_quad_logdet(mat.evaluate(), diff, logdet=True)
        return inv_quad
        
        
    def get_p21(self,g_theta1, g_theta2, data, model, noise_value):

        cov_noise =  noise_value * torch.eye(data.shape[0])    #likelihood._shaped_noise_covar([ agg_data.shape[0],
        #cov_noise2 =  noise_value * torch.eye(2 * g_theta2.shape[0])
        K = model.covar_module
        mu = model.mean_module
        C21 = K.forward(g_theta2, g_theta1)
        
        C11 = K.forward(g_theta1, g_theta1, add_jitter = True) + cov_noise
        
        device = g_theta1.device
        C11 = C11.to(device)
        C21 = C21.to(device)
        
        rhs_g = data - torch.flatten(mu.forward(g_theta1))
        rhs_g = rhs_g.reshape(rhs_g.shape, 1)
        
        mean2 = mu.forward(g_theta2)
      
        mean2 = torch.flatten(mean2)
       
        p21 = mean2.reshape(mean2.shape, 1) + C11.inv_matmul(rhs_g, C21.evaluate())
        return p21
        
    def get_pf12(self, Q21,g_theta1, g_theta2, x, data2, pf1,p21, model, noise_value):

        cov_noise =  noise_value * torch.eye(2 * g_theta1.shape[0])    #likelihood._shaped_noise_covar([ agg_data.shape[0],
        #cov_noise2 =  noise_value * torch.eye(2 * g_theta2.shape[0])
        device = g_theta1.device
        cov_noise = cov_noise.to(device)
        K = model.covar_module
        mu = model.mean_module
        C12 = K.forward(g_theta1, g_theta2)
        
        C11 = K.forward(g_theta1, g_theta1, add_jitter = True) + cov_noise
        
        Cx1 = K.forward(x, g_theta1)
        Cx2 = K.forward(x, g_theta2)
        
        rhs_g = data2 - p21
        rhs_g = rhs_g.reshape(rhs_g.shape[0], 1)
        
        lhs = Cx2 - C11.inv_matmul(C12.evaluate(), Cx1.evaluate())
       
        pf12 = pf1 + Q21.inv_matmul(rhs_g, lhs.evaluate())

        
        #p21 = p21.reshape(data.shape)
        
        
        
        return pf12

        
        

    
    
  
