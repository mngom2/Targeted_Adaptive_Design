from typing import Any

import math
import torch
from torch import nn
from torch import Tensor
import numpy as np
import os
import sys
from functools import partial
import gpytorch
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.distributions import base_distributions
from gpytorch.settings import max_cholesky_size
from matplotlib import pyplot as plt
sys.path.append("..")
import vvkernels as vvk
import sep_vvkernels as svvk
import vvk_rbfkernel as vvk_rbf
import vvmeans as vvm
import vvlikelihood as vvll
from vfield import VField

from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler







class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,num_base_kernels):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        a = torch.ones(2,2)
        chol_q = torch.tril(a)
        self.mean_module = vvm.TensorProductSubMean(gpytorch.means.ConstantMean(), num_tasks = 2)  # gpytorch.means.MultitaskMean(gpytorch.means.LinearMean(2), num_tasks = 2)     #vvm.TensorProductSubMean(gpytorch.means.LinearMean(2), num_tasks = 2)#vvm.TensorProductSubMean(gpytorch.means.ConstantMean(), num_tasks = 2)  #
        base_kernels = []
        for i in range(num_base_kernels):
            base_kernels.append(( (gpytorch.kernels.MaternKernel() ) )) #gpytorch.kernels.PolynomialKernel(4)  ##gpytorch.kernels.MaternKernel()# (vvk_rbf.vvkRBFKernel())
#         base_kernels2 = []
#         for i in range(num_base_kernels):
#             base_kernels2.append(gpytorch.kernels.PolynomialKernel(5))
            
        self.covar_module = svvk.SepTensorProductKernel(base_kernels,num_tasks = 2, rank = 2)
       # self.covar_module = gpytorch.kernels.LCMKernel(base_kernels,num_tasks = 2, rank =2)

#\         self.covar_module = gpytorch.kernels.MultitaskKernel(
#             gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
#         )
       # self.covar_module = vvk.TensorProductKernel(vvk_rbf.vvkRBFKernel(), a[0,0], a[1,0], a[1,1], num_tasks = 2, rank =1,  task_covar_prior=None)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
    






def hyper_opti(config,checkpoint_dir=None, data_dir_train = None, data_dir_val = None):
    
    hp_iter = 50
    
    
    likelihood = vvll.TensorProductLikelihood(num_tasks = 2)
    
    dataset_train = np.genfromtxt(data_dir_train, delimiter=',')
    dataset_val = np.genfromtxt(data_dir_val, delimiter=',')
    
    g_theta1 = Tensor(dataset_train[:, 0:2])
    train_y = Tensor(dataset_train[:, 2:4])
    agg_data = train_y.flatten()
    
    val_theta1 = Tensor(dataset_val[:, 0:2])
    val_train_y = Tensor(dataset_val[:, 2:4])
    val_agg_data = val_train_y.flatten()
    
    model = MultitaskGPModel(g_theta1, agg_data, likelihood, num_base_kernels = config["num_bk"])

    optimizer = torch.optim.Adam(model.parameters(),  lr=config["lr"])  # Includes GaussianLikelihood parameters
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    model.train()
    likelihood.train()
    for epoch in range(hp_iter):
        
        #for i in range( hp_iter):
        optimizer.zero_grad()
        output = model(g_theta1)
                #output_ll = likelihood(output)

                #loss = -likelihood.get_mll(agg_data,output_ll)
        loss = -mll(output, agg_data)
        
        loss.backward()

        #print('Iter %d/%d - Loss hyperparam: %.3f' % (i + 1, hp_iter, loss.item()))
        optimizer.step()
#        with torch.no_grad():
#            model.eval()
#            likelihood.eval()
#            predictions = likelihood(model(val_theta1))
#            val_loss = torch.norm((predictions.mean).flatten() - val_agg_data)
     
#
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model, likelihood), path)

        tune.report(loss=float(loss.detach().numpy()))

    print("Finished Training")



def main(num_samples=30, max_num_epochs=200):


    #data training and val
    
    data_dir_train = os.path.abspath("datasets/data_train")
    data_dir_val = os.path.abspath("datasets/data_val")
   
        
    ##################
    config = {
    "num_bk": tune.randint(1, 10),
    "lr": tune.loguniform(1e-4, 1.)
    }
    
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
        
    reporter = CLIReporter(
        parameter_columns=["num_bk", "lr"],
        metric_columns=["loss", "training_iteration"])
    result = tune.run(
        partial(hyper_opti, data_dir_train=data_dir_train, data_dir_val = data_dir_val),
        config=config,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)
    
    best_trial = result.get_best_trial("loss", "min", "last")
    print(best_trial)
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

#
    best_checkpoint_dir = best_trial.checkpoint.value
    model, likelihood = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
        
    return model, likelihood
   



if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=30, max_num_epochs=200)


    
    

        
        

    
    
  
