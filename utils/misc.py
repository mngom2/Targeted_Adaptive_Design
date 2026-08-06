# Support for creating the objective functions and the needed utility functions
#
# Marieme Ngom, ANL
#
"""
Support for simulating a vector field over a hypercube, with a known value at a
selected point.
"""

import numpy as np
import math
from scipy.stats import uniform
import torch
from torch import Tensor


######################################################################
######################################################################
######################################################################
class ObjFun(object):
    """
    the twin peaks problem, first example in our paper
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

    def get_gradient(self, x1_, x2_):
        d1J1 = -6. * (1. - x1_) * torch.exp(-x1_**2 - (x2_ + 1)**2) - 6. * x1_ * (1. - x1_)**2 * torch.exp(-x1_**2 - (x2_+1)**2)- 10 * (1./5. - 3. * x1_**2) * torch.exp(-x1_**2 - x2_ **2) + 20. * x1_ * (x1_/5 - x1_**3 - x2_**5)* torch.exp(-x1_**2 - x2_ **2)+ 6. * (x1_ + 2.) * torch.exp(- (x1_ + 2.)**2 - x2_ **2) + 1.
        d2J1 = -6. * (1. - x1_)**2 * (x2_ +1.) * torch.exp(-x1_**2 - (x2_ + 1)**2)+ 10 * (5 * x2_**4) * torch.exp(-x1_**2 - x2_ **2) + 20. * x2_ * (x1_/5 - x1_**3 - x2_**5)* torch.exp(-x1_**2 - x2_ **2) + 6. * (x2_) * torch.exp(- (x1_ + 2.)**2 - x2_ **2) + .5
        
        
        d1J1 = d1J1.reshape(d1J1.shape[0], 1)
        d2J1 = d2J1.reshape(d2J1.shape[0], 1)
                
        d1Vf_1 = d1J1/8.9280
        d2Vf_1 = d2J1/8.9280
        
        
        d1J2 = 6. * (-x1_+1) * (1.+x2_)**2 * torch.exp(-x2_**2 - (-x1_ +1.) ** 2)- 50 * x1_**4 * torch.exp(-x1_**2 - x2_ **2)  + 20 * x1_ * (-x2_/5. + x2_**3 + x1_**5) * torch.exp(-x1_**2 - x2_ **2)+ 6. * x1_ * torch.exp(- (2 - x2_)**2 -x1_**2)
                
        d2J2 = 6. * (1.+x2_) * torch.exp(-x2_**2 - (x1_ +1.) ** 2) - 6. * x2_ * (1 + x2_)**2 * torch.exp(-x2_**2 - (x1_ +1.) ** 2)- 10 * (-1./5. + 3 * x2_ **2) * torch.exp(-x1_**2 - x2_ **2)  + 20 * x2_ * (-x2_/5. + x2_**3 + x1_**5) * torch.exp(-x1_**2 - x2_ **2)- 6. * (2 - x2_) * torch.exp(- (2 - x2_)**2 -x1_**2) 
        
        d1J2 = d1J2.reshape(d1J2.shape[0], 1)
        d2J2 = d2J2.reshape(d2J2.shape[0], 1)
                
                
        d1Vf_2 = d1J2/8.9280
        d2Vf_2 = d2J2/8.9280
        
        d1Vf = d1Vf_1 + d1Vf_2
        d2Vf = d2Vf_2 + d2Vf_1
     
        return torch.cat((d1Vf_1, d2Vf_1, d1Vf_2, d2Vf_2), 1)

    def out_box(self,x):
        dim = x.shape[1]
        for i in range(dim):
    
            if any( y < self.low for y in x[i]) or any( y > self.high for y in x[i]):
                return True
            else:
                return False
        
######################################################################
######################################################################
class x2(object):
    """
    the twin peaks problem, first example in our paper
    """
#########################################
    def __init__(self, D=1, N = 1, low = -1., high = 1.,  tgt_loc=Tensor(np.array([[0.0]])),
                 tgt_vec = Tensor(np.array([0.]))):

        self.tgt_loc = tgt_loc
        self.tgt_vec= tgt_vec
        self.D = D
        self.N = N
        self.low = low
        self.high = high
    def _get_obj(self, x1_):
        out = x1_**2
        
        return out
    
    def __call__(self, x1_):
        return self._get_obj(x1_)

    def get_gradient(self, x1_):
        dout = 2. * x1_
     
        return dout


######################################################################
######################################################################
######################################################################


######################################################################
######################################################################
######################################################################
class Viennet(object):
    """
    Viennet function
    """
#########################################
    def __init__(self, D=2, N = 3, low = -3., high = 3.,  tgt_loc=None,
                 tgt_vec = Tensor(np.array([[0.9231,15.1532, 0.0307]]))):

        self.tgt_loc = tgt_loc
        self.tgt_vec= tgt_vec
        self.D = D
        self.N = N
        self.low = low
        self.high = high
    def _get_obj(self, x1_, x2_):

        J1 = 0.5 * (x1_**2 + x2_**2) + torch.sin(x1_**2 + x2_**2)  
        
        J2 = (3. * x1_ - 2. * x2_ + 4.)**2/8. + (x1_ - x2_ + 1)**2/27. + 15.
        
        J3 = 1. / (x1_**2 + x2_**2 + 1.) - 1.1 * torch.exp(-(x1_**2 + x2_**2))
        
        J1 = J1.reshape(J1.shape[0], 1)
        J2 = J2.reshape(J2.shape[0], 1)
        J3 = J3.reshape(J2.shape[0], 1)
        
        Vf_1 = J1/15.
        Vf_2 = J2/15.
        Vf_3 = J3 * 15.
        
        return torch.cat((Vf_1, Vf_2, Vf_3),1)
    
    def __call__(self, x1_, x2_):
        return self._get_obj(x1_, x2_)
        

######################################################################
######################################################################
######################################################################


class Viennet_Fonseca_Fleming(object):
    """
    Viennet function
    """
#########################################
    def __init__(self, D=2, N = 4, low = -3., high = 3.,  tgt_loc=None,
                 tgt_vec = Tensor(np.array([[0.9231,15.1532, 0.0307]]))):

        self.tgt_loc = tgt_loc
        self.tgt_vec= tgt_vec
        self.D = D
        self.N = N
        self.low = low
        self.high = high
    def _get_obj(self, x1_, x2_):

        J1 = 0.5 * (x1_**2 + x2_**2) + torch.sin(x1_**2 + x2_**2)  
        
        J2 = (3. * x1_ - 2. * x2_ + 4.)**2/8. + (x1_ - x2_ + 1)**2/27. + 15.
        
        J3 = 1. / (x1_**2 + x2_**2 + 1.) - 1.1 * torch.exp(-(x1_**2 + x2_**2))
        
        J4 = 1. - torch.exp( - (x1_ - 1./np.sqrt(2.))**2 - (x2_ - 1./np.sqrt(2.) )**2 )
       # J5 = 1. - torch.exp( - (x1_ + 1./np.sqrt(2.))**2 - (x2_ + 1./np.sqrt(2.) )**2 )
        
        J1 = J1.reshape(J1.shape[0], 1)
        J2 = J2.reshape(J2.shape[0], 1)
        J3 = J3.reshape(J3.shape[0], 1)
        J4 = J4.reshape(J4.shape[0], 1)
      #  J5 = J5.reshape(J5.shape[0], 1)
        
        Vf_1 = J1/15.
        Vf_2 = J2/15.
        Vf_3 = J3 * 15.
        Vf_4 = J4
       # Vf_5 = J5
        
        return torch.cat((Vf_1, Vf_2, Vf_3, Vf_4),1)
    
    def __call__(self, x1_, x2_):
        return self._get_obj(x1_, x2_)
        

######################################################################
######################################################################
######################################################################


class DTLZ1(object):
    """
    DTLZ1
    """
#########################################
    def __init__(self, D=7, N = 3, low = 0., high = 1.,  tgt_loc=None,
                 tgt_vec = None):

        self.tgt_loc = tgt_loc
        self.tgt_vec= tgt_vec
        self.D = D
        self.N = N
        self.k = self.D - self.N + 1
        self.low = low
        self.high = high
    def _get_obj(self, x_):
        n = self.N + (self.D - 2) - 1
        if (x_.shape[1] != n):
            print(x_.shape[1])
            print('sizez noyt consistent')
        x_M = x_[:, -(self.k):]

        g = 100. * ( self.k + torch.sum( (x_M - 0.5)**2 - torch.cos(20. * math.pi * (x_M -  0.5))))

        f = torch.zeros(x_.shape[0], self.N)
        f[:, 0] = 0.5 * x_[:,0] * x_[:,1] * (1. + g)
        f[:, 1] = 0.5 * x_[:,0] * (1. - x_[:,1]) * (1. + g)
        f[:, 2] = 0.5 * (1. - x_[:,0]) * (1. + g)

        return 1./1. * f.reshape(x_.shape[0],self.N)
    def out_box(self,x):
        dim = x.shape[1]
        for i in range(dim):

            if any( y < self.low for y in x[i]) or any( y > self.high for y in x[i]):
                return True
            else:
                return False
    
    def __call__(self, x_):
        return self._get_obj(x_)
        

######################################################################
######################################################################
######################################################################


class DTLZ4(object):
    """
    DTLZ4
    """
#########################################
    def __init__(self, D=4, N = 3, low = 0., high = 1.,  tgt_loc=None,
                 tgt_vec = None):

        self.tgt_loc = tgt_loc
        self.tgt_vec= tgt_vec
        self.D = D
        self.N = N
        self.k = self.D - self.N + 1
        self.low = low
        self.high = high
    def _get_obj(self, x_):
        alpha = 100.
        n = self.N + (self.D - 2) - 1
        if (x_.shape[1] != n):
            print(x_.shape[1])
            print('sizez noyt consistent')
        x_M = x_[:, self.N-1:]
        x_ = x_[:, :(self.N-1)]

        g =  torch.sum( (x_M - 0.5)**2 ) 
        coef_g = 1. + g

        f = torch.zeros(x_.shape[0], self.N).to(x_)
        for i in range(0, self.N):
            f[:,i] = coef_g * torch.prod( torch.cos( (x_[:,:x_.shape[1] - i])**alpha * math.pi/2. ) )
            if i>0:
                
                f[:,i] = f[:,i] * torch.sin( (x_[:,x_.shape[1] - i])**alpha * math.pi/2. )
        #f = torch.log(f)/10.                                        

        return 1./1. * f.reshape(x_.shape[0],self.N)
    
    
    def out_box(self,x):
        dim = x.shape[1]
        for i in range(dim):

            if any( y < self.low for y in x[i]) or any( y > self.high for y in x[i]):
                return True
            else:
                return False
    
    def __call__(self, x_):
        return self._get_obj(x_)
        

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

def stopping_criteria(tol_vector, pf1, f_target, lower_bound, upper_bound):
    """
    success criteria
    """
#########################################
    lower_tol_vector = f_target - tol_vector
    upper_tol_vector = f_target + tol_vector
    SUCCESS = True
    # print('$$$$$$$$$$$$$$$$$$')
    # print(lower_bound)
    # print(lower_tol_vector)
    # print(upper_bound)
    # print(upper_tol_vector)
    # print('$$$$$$$$$$$$$$$$$$')
    for i in range(f_target.shape[0]):
            if (lower_bound[i] < lower_tol_vector[i]) or  (upper_bound[i] > upper_tol_vector[i]):
                SUCCESS = False
            # if ((pf1[i]- f_target[i]) < lower_tol_vector[i]) or ((pf1[i]- f_target[i]) > upper_tol_vector[i]):
            #     SUCCESS = False
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


######################################################################
######################################################################
######################################################################



