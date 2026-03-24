"""
filename: para.py (adapted from aiyagari_penalty_func_12_28.py)
@authors: Zhouzhou Gu, Jonathan Payne

This file contains parameters, and basic functions used in the model

(Parent Class)  (Subclass)
Environment - > Household
            - > Trainer_pde
            - > Traning Sampler
"""

import numpy as np
import torch

## --------------------------------------------------
## A. Parameters
## --------------------------------------------------
# Put things into a dictionary, and then form the Environment Class
params = {
    "gam": 2.1,     # crra preference's parameter
    "dep": 0.1,     # depreciation
    "rho": 0.05,    # discount factor
    "lam1": 0.4,    # income jump rate 2->1
    "lam2": 0.4,    # income jump rate 1->2
    "y0": 0.3,      # the low labor income state
    "alp":1/3,      # Production fn power: Y=Zprod*K^alp*L^(1-alp)
    "Zprod":1.0,    # Productivity in: : Y=Zprod*K^alp*L^(1-alp)
    "eta": 0.5,     # reverting process's parameter
    "sig": 0.01,    # productivity volatility
    "a_min": 1e-2,  # minimum asset amount 1e-2
    "a_max": 20.0,  # maximum asset amount 20.0 (can be increase to 35.0)
    "a_max_FD":21.0, # upper bound on net assets (for fd solution)
    "N_FD": 1000,   # number of finite difference girds
    "a_lb": 1.0,    # from penalty kappa/2*max(-a+a_lb,0)**2
    "kappa":3.0,    # from penalty kappa/2*max(-a+a_lb,0)**2
    "Zprod_min":0.96, # Maximum of the production factor
    "Zprod_max":1.04, # Minimum of the production factor
    "r":0.02,       # initial guess of interest rate (for fd solution)
    "w":1.0,        # initial guess of wage rate (for fd solution)
    "crit_fd":1e-6, # convergence criteria for finite difference
    "r_lb":-0.05,   # minimum interest rate (when varying avg. K)
    "r_rb":0.1,     # maximum interest rate (when varying avg. K)
    "n_pop":41,     # number of agents
    "nn_width":64,  # network width per layer
    "nn_num_layers":5, # number of layers
    "Num_intervals":2**4,    # Number of partitions     
    "Num_sample_points":2**8,    # Number of sampling points in each partition
    "Num_batch_points":2**4,    # Number of batch points in each partition
    "num_of_al_points":2**10,        # points in active learning region
    "num_of_al_points_batch":2**4,   # points in active learning region, batch size
    "num_of_bc_points":2**10,        # points on the boundary
    "num_of_bc_points_batch":2**0,   # points on the boundary, batch size
    "num_additional_points":2**4,    # points add to where the res_loss is largest
    "n_batch":2**4                  # number of batches
}

params["y_vals"] = [params["y0"],1+params["lam2"]/params["lam1"]*(1-params["y0"])]

class Environments():
    def __init__(self,params):
        self.gam = params["gam"]
        self.dep = params["dep"]
        self.rho = params["rho"]
        self.lam1 = params["lam1"]
        self.lam2 = params["lam2"]
        self.lams = np.array([[-self.lam1, self.lam1], [self.lam2, -self.lam2]])
        self.lamsd = np.array([self.lam1, self.lam2])
        self.PI = [[-self.lam1, self.lam1], [self.lam2, -self.lam2]]
        self.y_vals = [params["y0"],1+self.lam2/self.lam1*(1-params["y0"])]
        self.alp = params["alp"]
        self.Zprod = params["Zprod"]
        self.Zprod_max = params["Zprod_max"]
        self.Zprod_min = params["Zprod_min"]
        self.y_ave =(self.y_vals[0]*self.lams[0, 1] \
                    + self.y_vals[1]*self.lams[1, 0])\
                    / (self.lams[0, 1] + self.lams[1, 0])
        self.a_min = params["a_min"]
        self.a_max = params["a_max"]
        self.a_max_FD = params["a_max_FD"]
        self.N_FD = params["N_FD"]
        self.a_lb = params["a_lb"]
        self.kappa = params["kappa"]
        self.crit_fd = params["crit_fd"]
        self.n_pop = params["n_pop"]
        self.eta = params["eta"]
        self.sig = params["sig"]
        self.r_rb = params["r_rb"]
        self.r_lb = params["r_lb"]

    ## --------------------------------------------------
    ## D. Functions
    ## --------------------------------------------------
    def u(self,c):
        """Agent utility function.
        """
        return c**(1-self.gam)/(1-self.gam)
    def du(self,c):
        """Derivative of agent utility function.
        """
        return c**(-1*self.gam)

    def rental(self,K,L,Z):
        """Rental rate on capital (given K and L)
        """
        return Z*self.alp*(L/K)**(1-self.alp)-self.dep

    def wage(self,K,L,Z):
        """Wage rate on labor (given K and L)
        """
        return Z*(1-self.alp)*(K/L)**self.alp

    def Kap(self, a, n_pop):
        """Aggregate capital stock
        """
        return torch.sum(a, 1)/n_pop
        #return torch.sum(a, 1)

    def Lab(self, y, n_pop):
        """Aggregate labor
        """
        yis = torch.sum(y, 1)
        return (self.y_vals[1]*yis + self.y_vals[0]*(n_pop - yis))/n_pop
        #return (y_vals[1]*yis + y_vals[0]*(n_pop - yis))

    def k_inv(self, r, Z):
        """Calculate capital stock for a given interest rate.
        """
        return (Z*self.alp/(r+self.dep))**(1/(1-self.alp))
    
    def calc_eqm(self, x,Z,k_ss = 5.0 ,r_ss = 0.02 ,w_ss = 1.0):
        """Calculates the equilibrium (without tracking gradient).
        """
        if self.n_pop == 1:
            K = k_ss
            L = self.Lab(x[:,1:self.n_pop], self.n_pop-1)
            r = r_ss
            w = w_ss
        else:
            K = self.Kap(x[:,1:self.n_pop], self.n_pop-1)
            L = self.Lab(x[:,self.n_pop+1:], self.n_pop-1)
            r = self.rental(K,L,Z)
            w = self.wage(K,L,Z)
        ## Return
        return K, L, r, w
