"""
filename: aiy_fd_crra.py
@author: Jonathan Payne (adapted from Moll's website).
"""

## This file contains sample code for solving the steady state for
## for an Aiyagari model using finite difference
##
## Solves the Aiyagari model with CRRA utility function:
##  𝜌V(a,y) = u(c^*) + ∂/∂a V(a,y)(wy + ra - c^*) + λ (V(a,y') - V(a,y))
## s.t.
##  u(c) = c^(1-gam)/(1-gam)
##  c^*(a,y) = (u')^(-1)(∂/∂a V(a,y)) = (∂/∂a V(a,y))^(-1/𝛾)
##  ∂/∂a V(abar,y) ≥ u'(y + ra) = (y + ra)^(-1/𝛾)
##
## Technique:
##   - Uses finite difference
##
## To run:
##   ipython
##   %run aiy_fd_crra.py

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from para import *

## --------------------------------------------------
## Household Class
## --------------------------------------------------
class Household(Environments):
    def __init__(self, params, a_size = 1000,delta = 100.0):
        ''' Household class is the subclass  of Environments
            a_size = 1000,     number of asset grid points
            delta  = 1000.0,   1/delta is time step in fd iteration
        '''
        # Initialize values, and set up grids over a and z
        super().__init__(params)
        self.gam    = params["gam"]
        self.r, self.w,  = params["r"], params["w"]
        self.rho, self.dep = params["rho"], params["dep"]
        self.a_min, self.a_max,  = params["a_min"], params["a_max"]
        self.a_size = a_size
        self.da     = (self.a_max-self.a_min)/(self.a_size-1)
        self.k      = 10
        self.pi     = np.asarray(self.PI)
        self.z_vals = np.asarray(self.y_vals)
        self.z_size = len(self.y_vals)
        self.a_vals = np.linspace(self.a_min, self.a_max, self.a_size)
        self.n      = self.a_size * self.z_size
        self.delta  = delta
        self.kappa  = params["kappa"]
        self.a_lb   = params["a_lb"]
        self.alp    = params["alp"]
        self.Zprod  = params["Zprod"]

        ###### ADDED TO MATCH LABOR SUPPLY IN .m
        self.z_ave  = (self.z_vals[0]*self.pi[0, 1] +
                        self.z_vals[1]*self.pi[1, 0]) / \
                        (self.pi[0, 1] + self.pi[1, 0])

        # Initial Guess of Value Function
        self.v      = np.log(np.tile(self.a_vals,(self.z_size,1))*self.r
                        +self.w*np.tile(self.z_vals,(self.a_size,1)
                            ).transpose())/self.rho

        # Build skill_transition, the matrix summarizing transitions due to the
        # Poisson income shocks. This is analogous to the Q matrix in the
        # discrete time version of the QuantEcon Aiyagari model
        self.z_transition = sparse.kron(self.pi,sparse.eye(self.a_size),
                                        format="csr")

        # Preallocation
        self.v_old  = np.zeros((self.z_size,self.a_size))
        self.g      = np.zeros((self.z_size,self.a_size))
        self.dv     = np.zeros((self.z_size,self.a_size-1))
        self.cf     = np.zeros((self.z_size,self.a_size-1))
        self.c0     = np.zeros((self.z_size,self.a_size))
        self.ssf    = np.zeros((self.z_size,self.a_size))
        self.ssb    = np.zeros((self.z_size,self.a_size))
        self.is_forward = np.zeros((self.z_size,self.a_size),'bool')
        self.is_backward = np.zeros((self.z_size,self.a_size),'bool')
        self.diag_helper = np.zeros((self.z_size,self.a_size))
        self.A      = self.z_transition.copy()
        self.B      = self.z_transition.copy()
        self.AT     = self.z_transition.copy()
        self.w_ss   = 0.05
        self.k_ss   = 1.0
        self.r_ss   = 0.05


    def set_prices(self, r, w):
        """
        Resets prices
        Calling the method will resolves the Bellman Equation.

        Parameters:
        -----------------
        r : Interest rate
        w : 
        """
        self.r, self.w = r, w
        self.solve_bellman()


    def reinitialize_v(self):
        """
        Reinitializes the value function if the value function
        became NaN
        """
        self.v = np.log(np.tile(self.a_vals,(self.z_size,1))*self.r
                        +self.w*np.tile(self.z_vals,(self.a_size,1)
                            ).transpose())/self.rho

    def penalty_v(self):
        """
        The penalty function: the quadratic case
        """
        # you can also use (a-a_lb)^4 to penalize, which is smoother
        return -self.kappa/2*np.maximum(-self.a_vals+self.a_lb,0)**2
    
    def solve_bellman(self,maxiter=100,crit=1e-6):
        """
        This function solves the decision problem with the given parameters

        Parameters:
        -----------------
        maxiter :   maximum number of iteration before haulting
                    value function iteration

        crit :      convergence metric, stops if value function does not
                    change more than crit
        """
        dist=100.0
        for i in range(maxiter):
            # compute saving and consumption implied by current guess for
            # ...value function, using upwind method
            self.dv = (self.v[:,1:]-self.v[:,:-1])/self.da
            self.cf = (self.dv)**(-1/self.gam)
            self.c0 = np.tile(self.a_vals,(self.z_size,1))*self.r \
                        +self.w*np.tile(self.z_vals,(self.a_size,1)).transpose()

            # computes savings with forward and backward difference
            self.ssf[:,:-1] = self.c0[:,:-1]-self.cf
            self.ssb[:,1:] = self.c0[:,1:]-self.cf
            # Note that the boundary conditions are handled implicitly
            # ...as ssf will be zero at a_max and ssb at a_min
            self.is_forward = self.ssf>0
            self.is_backward = self.ssb<0
            # Update consumption based on forward or backward difference based
            # ...on direction of drift
            self.c0[:,:-1] += (self.cf-self.c0[:,:-1])*self.is_forward[:,:-1]
            self.c0[:,1:] += (self.cf-self.c0[:,1:])*self.is_backward[:,1:]
            ######
            self.c0 = (self.c0)**(1-self.gam)/(1-self.gam)
            # dynamic method, pass one variable
            self.c0 = self.c0 + self.penalty_v()
            # Build the matrix A that summarizes the evolution of the process
            # ...for (a,z). This is a Poisson transition matrix
            # ...(aka intensity matrix) with rows adding up to zero
            self.A = self.z_transition.copy()
            self.diag_helper = (-self.ssf*self.is_forward/self.da \
                               + self.ssb*self.is_backward/self.da
                                    ).reshape(self.n)
            self.A += sparse.spdiags(self.diag_helper,0,self.n,self.n)
            self.diag_helper = (-self.ssb*self.is_backward/self.da
                                    ).reshape(self.n)
            self.A += sparse.spdiags(self.diag_helper[1:],-1,self.n,self.n)
            self.diag_helper = (self.ssf*self.is_forward/self.da
                                    ).reshape(self.n)
            self.A += sparse.spdiags(np.hstack((0,self.diag_helper)),1,
                                    self.n,self.n)
            # Solve the system of linear equations corresponding to implicit
            # ...finite difference scheme
            self.B = sparse.eye(self.n)*(1/self.delta + self.rho) - self.A
            self.b = (self.c0.reshape(self.n,1)
                        + self.v.reshape(self.n,1)/self.delta)
            self.v_old = self.v.copy()
            self.v = spsolve(self.B,self.b).reshape(self.z_size,self.a_size)

            # Compute convergence metric and stop if it satisfies the
            # ...convergence criterion
            dist = np.amax(np.absolute(self.v_old-self.v).reshape(self.n))
            if dist < crit:
                break

    def compute_stationary_distribution(self):
        """
        Solves for the stationary distribution given household decision rules

        Output:
        Capital level from the stationary distribution
        """
        self.AT = self.A.transpose().tocsr()

        # The discretized Kolmogorov Forward equation AT*g=0 is an eigenvalue
        # ...problem. AT is singular because one of the equation is the
        # ...distribution adding up to 1. Here we solve the eigenvalue problem
        # ...by setting g(1,1)=0.1 and the equation is solved relative to that
        # ...value. Alternatively, one could use a routine for solving
        # ...eigenvalue problems.
        b = np.zeros((self.n,1))
        b[0] = 0.1
        self.AT.data[1:self.AT.indptr[1]] = 0
        self.AT.data[0] = 1.0
        self.AT.indices[0] = 0
        self.AT.eliminate_zeros()
        self.g = spsolve(self.AT,b).reshape(self.z_size,self.a_size)

        # Since g was solved taking one of g(1,1) as given, g needs to be
        # renormalized to add up to 1
        self.g = self.g/np.sum(self.g)
        return np.sum(self.g*(np.tile(self.a_vals,(self.z_size,1))))

    def r_to_w(self, r):
        """Caculate wage rate for a given interest rate.
        """
        return self.Zprod*(1-self.alp) * \
            (self.alp*self.Zprod/(self.dep+r))**(self.alp/(1-self.alp))

    def rd(self, K):
        """Calculate interest rate given capital stock.
        """
        return self.Zprod*self.alp*(self.z_ave/K)**(1-self.alp)-self.dep

    def k_inv(self,r):
        """Calculate capital stock for a given interest rate.
        """
        return (self.Zprod*self.alp/(r+self.dep))**(1/(1-self.alp))

    def bisection_loop(self):
        """
        Solves the equilibrium interest rate, wage and distribution
        """
        # Bisection loop
        r_min   = 0.00          # minimum interest rate for bisection
        r_max   = 0.05          # maximum interest rate for bisection
        r_ss    = 0.03          # Initialize interest rate
        w_ss    = self.r_to_w(r_ss)
        crit_fd = 1e-6          # convergence criteria for finite difference
        for i in range(100):
            self.set_prices(r_ss,self.r_to_w(r_ss))
            r_new = self.rd(self.compute_stationary_distribution())
            if np.absolute(r_new-r_ss)<crit_fd:
                break
            elif r_new > r_ss:
                r_min = r_ss
                r_ss = (r_max+r_min)/2.
            else:
                r_max = r_ss
                r_ss = (r_max+r_min)/2.

        w_ss = self.r_to_w(r_ss)
        k_ss = self.k_inv(r_ss)
        self.w_ss = w_ss
        self.k_ss = k_ss
        self.r_ss = r_ss
        self.g = np.abs(self.g) # numerically, g can be slightly negative ~10^-20
