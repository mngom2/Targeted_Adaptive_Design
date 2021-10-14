from typing import Any, Optional
import warnings
from copy import deepcopy
import math
import torch
from torch import nn
from torch import Tensor
import gpytorch
from gpytorch.likelihoods import MultitaskGaussianLikelihood, _MultitaskGaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import MultitaskHomoskedasticNoise, FixedGaussianNoise, Noise
from gpytorch.distributions import base_distributions
from gpytorch.settings import max_cholesky_size
from gpytorch.lazy import ZeroLazyTensor, DiagLazyTensor
from gpytorch.utils.warnings import GPInputWarning
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
       

        
        cov_noise1 = likelihood._shaped_noise_covar([g_theta1.shape[0],g_theta1.shape[0]]).evaluate()
        cov_noise2 = likelihood._shaped_noise_covar([g_theta2.shape[0],  g_theta2.shape[0]]).evaluate()

         
        mu = model.mean_module
        K = model.covar_module


        Cff = K.forward(x,x)
       
        Cf1 = K.forward(x,g_theta1)
        C11 = K.forward(g_theta1, g_theta1) #+ cov_noise1
     
        Cf2 = K.forward(x, g_theta2)
        
        C12 = K.forward(g_theta1, g_theta2)
        C22 = K.forward(g_theta2, g_theta2) #+ cov_noise2 #+ 0.005 * torch.eye(cov_noise2.shape[0])     #cov_noise2
    
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
        Q21 = Q21.add_jitter(1e-8)

        
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
        
            for param_name, param in model.named_parameters():
                print(param_name)
                print(param)
            raise
        
        Qf12 = Qf1 - Qf12_sec_term
        Qf12 = Qf12.add_jitter(1e-8)
        
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





    def get_pll(self, f_target, x,g_theta, agg_data, model, likelihood):

        """
        computes the predicted ll needed for the first iteration of the algorithm

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

        #pll = 1./2. * (logdet_Q + inv_quad_Q)
        pll = torch.linalg.norm((f_target - m), ord = 2)  #
        lower_bound = torch.zeros(m.shape)
        upper_bound = torch.zeros(m.shape)
       
        for i in range(m.shape[0]):
            lower_bound[i] = m[i] -  torch.sqrt(Q[i,i])
            upper_bound[i] = m[i] +  torch.sqrt(Q[i,i])

        return pll, lower_bound, upper_bound
        
        
        
class FixedNoiseMultitaskGaussianLikelihood(_MultitaskGaussianLikelihoodBase):
    """
    A Likelihood that assumes fixed heteroscedastic noise. This is useful when you have fixed, known observation
    noise for each training example.

    Note that this likelihood takes an additional argument when you call it, `noise`, that adds a specified amount
    of noise to the passed MultivariateNormal. This allows for adding known observational noise to test data.

    .. note::
        This likelihood can be used for exact or approximate inference.

    :param noise: Known observation noise (variance) for each training example.
    :type noise: torch.Tensor (... x N)
    :param learn_additional_noise: Set to true if you additionally want to
        learn added diagonal noise, similar to GaussianLikelihood.
    :type learn_additional_noise: bool, optional
    :param batch_shape: The batch shape of the learned noise parameter (default
        []) if :obj:`learn_additional_noise=True`.
    :type batch_shape: torch.Size, optional

    :var torch.Tensor noise: :math:`\sigma^2` parameter (noise)

    Example:
        >>> train_x = torch.randn(55, 2)
        >>> noises = torch.ones(55) * 0.01
        >>> likelihood = FixedNoiseGaussianLikelihood(noise=noises, learn_additional_noise=True)
        >>> pred_y = likelihood(gp_model(train_x))
        >>>
        >>> test_x = torch.randn(21, 2)
        >>> test_noises = torch.ones(21) * 0.02
        >>> pred_y = likelihood(gp_model(test_x), noise=test_noises)
    """

    def __init__(
        self,
        num_tasks,
        noise: Tensor,
        learn_additional_noise: Optional[bool] = False,
        batch_shape: Optional[torch.Size] = torch.Size(),
        **kwargs: Any,
    ) -> None:
        super().__init__(num_tasks = num_tasks, noise_covar=FixedGaussianNoise(noise=noise))
        self.has_global_noise=False
        if learn_additional_noise:
            noise_prior = kwargs.get("noise_prior", None)
            noise_constraint = kwargs.get("noise_constraint", None)
            self.second_noise_covar = MultitaskHomoskedasticNoise(
                noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape, num_tasks= num_tasks
            )
        else:
            self.second_noise_covar = None

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise + self.second_noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def second_noise(self) -> Tensor:
        if self.second_noise_covar is None:
            return 0
        else:
            return self.second_noise_covar.noise

    @second_noise.setter
    def second_noise(self, value: Tensor) -> None:
        if self.second_noise_covar is None:
            raise RuntimeError(
                "Attempting to set secondary learned noise for FixedNoiseGaussianLikelihood, "
                "but learn_additional_noise must have been False!"
            )
        self.second_noise_covar.initialize(noise=value)

    def get_fantasy_likelihood(self, **kwargs):
        if "noise" not in kwargs:
            raise RuntimeError("FixedNoiseGaussianLikelihood.fantasize requires a `noise` kwarg")
        old_noise_covar = self.noise_covar
        self.noise_covar = None
        fantasy_liklihood = deepcopy(self)
        self.noise_covar = old_noise_covar

        old_noise = old_noise_covar.noise
        new_noise = kwargs.get("noise")
        if old_noise.dim() != new_noise.dim():
            old_noise = old_noise.expand(*new_noise.shape[:-1], old_noise.shape[-1])
        fantasy_liklihood.noise_covar = FixedGaussianNoise(noise=torch.cat([old_noise, new_noise], -1))
        return fantasy_liklihood

    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any):
        if len(params) > 0:
            # we can infer the shape from the params
            shape = None
        else:
            # here shape[:-1] is the batch shape requested, and shape[-1] is `n`, the number of points
            shape = base_shape

        res = self.noise_covar(*params, shape=shape, **kwargs)

        if self.second_noise_covar is not None:
            res = res + self.second_noise_covar(*params, shape=shape, **kwargs)
        elif isinstance(res, ZeroLazyTensor):
            warnings.warn(
                "You have passed data through a FixedNoiseGaussianLikelihood that did not match the size "
                "of the fixed noise, *and* you did not specify noise. This is treated as a no-op.",
                GPInputWarning,
            )

        return res
        
        
    def get_ell(self, agg_data, f_target, x, g_theta1, g_theta2, model, likelihood): #, cov_noise1, cov_noise2):

        """
        computes the expected ll

        """
       

        
        cov_noise1 =  0.0001 * torch.eye(agg_data.shape[0])     #likelihood._shaped_noise_covar([g_theta1.shape[0],g_theta1.shape[0]]).evaluate()
        cov_noise2 =  0.0001 * torch.eye(2 * g_theta2.shape[0])     #likelihood._shaped_noise_covar([g_theta2.shape[0],  g_theta2.shape[0]]).evaluate()
#print(cov_noise1)
         
        mu = model.mean_module
        K = model.covar_module


        Cff = K.forward(x,x)
       
        Cf1 = K.forward(x,g_theta1)
        C11 = K.forward(g_theta1, g_theta1) +cov_noise1
     
        Cf2 = K.forward(x, g_theta2)
        
        C12 = K.forward(g_theta1, g_theta2)
        C22 = K.forward(g_theta2, g_theta2) + cov_noise2  #   cov_noise2 #+ 0.005 * torch.eye(cov_noise2.shape[0])     #cov_noise2
    
        mean_data = mu.forward(g_theta1)
        
        mean_data = torch.flatten(mean_data)    #mean_data.reshape(mean_data.shape[0] * mean_data.shape[1], 1)
        #agg_data = agg_data.reshape(mean_data.shape[0], 1)
       # noise1_vec = torch.sqrt(cov_noise1.diag())
        rhs_g1 =  agg_data  - mean_data #.reshape(mean_data.shape, 1) #g-S #+ noise1_vec #+ 0.005 * torch.ones(agg_data.shape)
        
        
        
        mean_x = mu.forward(x)
      
        mean_x = torch.flatten(mean_x)  #mean_x.reshape(mean_x.shape[0] * mean_x.shape[1], 1)
        
        pf1 = mean_x.reshape(mean_x.shape, 1) + C11.inv_matmul(rhs_g1, Cf1.evaluate())
         
        C21 = C12.t()
        Q21 = C22 - C11.inv_matmul(C12.evaluate(), C21.evaluate())
       # Q21 =  Q21.add_jitter(1e-8)
        
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
        
            for param_name, param in model.named_parameters():
                print(param_name)
                print(param)
            raise
        
        Qf12 = Qf1 - Qf12_sec_term
        #l;Qf12 = Qf12.add_jitter(1e-8)
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



    

        
        

    
    
  
