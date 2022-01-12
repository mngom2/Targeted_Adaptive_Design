# Support for creating the objective functions and the needed utility functions
#
# Marieme Ngom, ANL
#
"""
Support for simulating a vector field over a hypercube, with a known value at a
selected point.
"""

import numpy as np
from scipy.stats import uniform
import torch
from torch import Tensor


######################################################################
######################################################################
######################################################################
class ObjFun(object):
    """
    An N-dimensional vector field over a D-dimensional hypercube, with
    a given value at a given point.
    """
#########################################
    def __init__(self, D=2, N = 2, low = -3., high = 3.,  tgt_loc=Tensor(np.array([[0.8731,0.5664]])),
                 tgt_vec = Tensor(np.array([3.0173/8.9280,3.1267/8.9280]))):

        self.tgt_loc = tgt_loc
        self.tgt_vec= tgt_vec
        self.D = D
        self.N = N
        self.low = low
        self.high = high
    def _get_obj(self, x1_, x2_):

        J1 = 3.* (1. - x1_) ** 2. * torch.exp(- x1_ **2 - (x2_ + 1) **2 ) - 10. * (x1_/5. - x1_ **3 - x2_**5)*torch.exp(-x1_**2 - x2_ **2) - 3 * torch.exp(- x2_ **2 - (x1_ + 2) **2 ) + 0.5 * (2 * x1_ + x2_)
        
        
        
        J2 = 3. * (1. + x2_) ** 2. * torch.exp(-x2_ **2 - (-x1_ + 1) **2 ) - 10. * (-x2_/5. + x2_ **3 + x1_**5)*torch.exp(-x2_**2 - x1_ **2) - 3. * torch.exp(- x1_ **2 - (-x2_ + 2) **2 ) #+ 0.5 * (-2 * x2_+x1_)
        J1 = J1.reshape(J1.shape[0], 1)
        J2 = J2.reshape(J2.shape[0], 1)
        
        Vf_1 = J1/8.9280
        Vf_2 = J2/8.9280
        
        return torch.cat((Vf_1, Vf_2),1)
    
    def __call__(self, x1_, x2_):
        return self._get_obj(x1_, x2_)
        

######################################################################
######################################################################
######################################################################

def get_vertices(center, radius_x, radius_y):

    """
    vertices of the TTR box
    """
#########################################
    x_1 = center[0] - radius_x
    y_1 = center[1] - radius_y
    
    x_2 = center[0] + radius_x
    y_2 = center[1] + radius_y
    
    x_3 = center[0] - radius_x
    y_3 = center[1] + radius_y
    
    x_4 = center[0] + radius_x
    y_4 = center[1] - radius_y
    
    v1 = Tensor([x_1, y_1]).reshape(2,1)
    v2 = Tensor([x_2, y_2]).reshape(2,1)
    v3 = Tensor([x_3, y_3]).reshape(2,1)
    v4 = Tensor([x_4, y_4]).reshape(2,1)
    
    return v1, v2, v3, v4


######################################################################
######################################################################
######################################################################

def stopping_criteria(tol_vector, f_target, lower_bound, upper_bound):
    """
    success criteria
    """
#########################################
    lower_tol_vector = f_target - tol_vector
    upper_tol_vector = f_target + tol_vector
    SUCCESS = True
    for i in range(f_target.shape[0]):
            if (lower_bound[i] < lower_tol_vector[i]) or  (upper_bound[i] > upper_tol_vector[i]):
                SUCCESS = False
    return SUCCESS
    

######################################################################
######################################################################
######################################################################
    
def check_dist(y1, y2,tol):
    """
    distance between 2 points
    """
#########################################
    index = range(y2.shape[0])
           
    index_del = []
    check = False
    for ii in range (y1.shape[0]):
    #         if (x[ii,0] < 0 or x[ii,1] <0):
    #             check = True
    #             index_del.append(ii)
        for jj in range(y2.shape[0]):
            if (torch.norm(y1[ii] - y2[jj])) <= tol:
                check = True
                index_del.append(jj)
        if check == True :
            index_del = (np.unique(index_del,))
            index_del = np.array((index_del), dtype = int)
                    
            index_ = np.delete(index, index_del)
            index_ = np.array(index_)
            y2_new = torch.zeros(index_.shape[0], y2.shape[1])
            jj = 0
            for ii in index_:
                        
                y2_new[jj] = y2[ii]
                jj = jj + 1
        else:
            index_ = index
            y2_new = y2
        return y2_new, check
        

######################################################################
######################################################################
######################################################################

def filter_sample(x,tol):
    
    """
    if sampling points get too clse to each other, drop them
    """
#########################################
    index = range(x.shape[0])
       
    index_del = []
    check = False
    for ii in range (x.shape[0]):
#         if (x[ii,0] < 0 or x[ii,1] <0):
#             check = True
#             index_del.append(ii)
        for jj in range(ii+1, x.shape[0]):
            if (torch.norm(x[ii] - x[jj])) <= tol:
                check = True
                index_del.append(jj)
    if check == True :
        index_del = (np.unique(index_del,))
        index_del = np.array((index_del), dtype = int)
                
        index_ = np.delete(index, index_del)
        index_ = np.array(index_)
        x_new = torch.zeros(index_.shape[0], x.shape[1])
        jj = 0
        for ii in index_:
                    
            x_new[jj] = x[ii]
            jj = jj + 1
    else:
        index_ = index
        x_new = x
    return x_new, check
