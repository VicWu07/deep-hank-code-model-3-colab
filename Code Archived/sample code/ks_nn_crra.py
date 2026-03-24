"""
filename: ks_nn_crra.py
@authors: Zhouzhou Gu, Jonathan Payne

This file contains the classes for solving the Krusell-Smith model 
using neural networks. 

Formally, solves the following pde for the V(aⁱ,yⁱ,z,s) = ∂/∂a V(a,y,z,s) 
(the derivative of the value function):
    𝜌V(aⁱ,yⁱ,z,s) = ∂/∂aⁱ V(aⁱ,yⁱ,z,s)(wyⁱ + raⁱ - c(aⁱ,yⁱ,z,s)) + V(aⁱ,yⁱ,z,s)r
                + λ(yⁱ)(V(aⁱ,(yⁱ)ᶜ,z,s) - V(aⁱ,yⁱ,z,s))
                + ∂/∂z V(aⁱ,yⁱ,z,s)μ(z) + 0.5σ²∂/∂z² V(aⁱ,yⁱ,z,s) 
                + Σ_{j≠i} ∂/∂aʲ V(aⁱ,yⁱ,zs)(wyʲ + raʲ - c(aʲ,yʲ,zs))
                + Σ_{j≠i} (V(aⁱ,yⁱ,z,s|(yʲ)ᶜ) - V(aⁱ,yⁱ,z,s|yʲ))
s.t.
    s = ({aʲ,yʲ}_{j≠i})
    (y)ᶜ = complement of y
    u(c) = c^(1-gam)/(1-gam)
    c^*(aⁱ,yⁱ,s) = (u')^(-1)(V(aⁱ,yⁱ,s)) = (V(aⁱ,yⁱ,s))^(-1/𝛾)
    V(amin,yⁱ,s) ≥ u'(yⁱ + r*amin) = (y + r*amin)^(-1/𝛾)

The file contains two main classes:
    - Training_pde: contains the pde operator
    - Training_Sampler: contains the sampling methods


"""
import torch
import numpy as np
from pyDOE import lhs

from para import *
from aiy_fd_crra import Household
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## --------------------------------------------------
## A. Preliminary functions
## --------------------------------------------------
class Training_pde(Environments):

    def __init__(self, params):
        super().__init__(params)
    
    def get_derivs_1order(self, y_pred, x):
        """ Returns zeroth, first and second derivatives.
            Uses automatic differentation to take fall derivatives.
        """
        dy_dx = torch.autograd.grad(y_pred, x,
                            create_graph=True,
                            grad_outputs=torch.ones_like(y_pred))[0]
        return dy_dx ## Return 'automatic' gradient.

    def Psi_a(self, a):
        """First order derivative of penalty function:
            dpsi/da, psi = -1/2 kappa (a-a_lb)^2 , a<a_lb
        """
        m = torch.nn.ReLU()
        return torch.tensor(self.kappa)*m(self.a_lb-a)
        
    def dvdz_oper(self,model,x,Z):
        
        z = Z.clone()
        z.requires_grad_(True)
        def model_z(z):
            return model(x, z)

        Vz_pred = model_z(z)
        dV_dz = self.get_derivs_1order(Vz_pred, z)
        return dV_dz

    ## --------------------------------------------------
    ## B. PDE
    ## --------------------------------------------------
    def pde_oper(self, model, x, Z):
        """ Constructs the pde operator, ℒ, in the pde 
            (master equation's derivative w.r.t a^i):
                ℒ u(x) - g(x) = 0
                ℒ u(x) = sum_i [mu_ai * partial u/partial a_i 
                            + lambda * u(switch yi)]
            Inputs
                model   = neural network
                x       = sample data for training
                x's size        : (Num_sample,2*n_pop)
                x[:,0:n_pop]    : asset holdings of agent 1 to agent n_pop
                x[:,n_pop+1:end]: employment status of agent 1 to agent n_pop
            General Idea:
                1.  Construct equilibrium from each agent j's perspective
                    and find consumption policy
                2.  Construct the master equation term by term and sum together
        """
        ## Extract varaibles and set up functions
        a = x[:,0:self.n_pop].clone()        # Extract "a" training values
        
        z = Z.clone()
        a.requires_grad_(True)          # Inititiate auto. diff. tracking
        
        def model_a(a):
            """ Set up neural network to differentiate w.r.t. a^1...a^n_pop
                    (no need to auto diff w.r.t y)
                a: the asset holding position from agent 1 to n_pop
            """
            x_temp = torch.clone(x)     # Clone data
            x_temp[:,0:self.n_pop] = a       
            return model(x_temp,z)

        ## Calculate equilibrium 
        K, L, r, w = self.calc_eqm(x,z)

        ## ------------------------------
        ## Compute the differential equation
        ## ------------------------------
        diffeq  = 0.0        # Initialize

        ## Terms at (a^1,y^1)
        ## ------------------------------
        ## Pass through NN to get current value function and derivatives
        V       = model(x,z)
        Va_pred = model_a(a)
        dV_da   = self.get_derivs_1order(Va_pred, a)
        
        ## Solve for household policies from equation u'(c) = V'
        ai      = x[:,0]
        yi      = x[:,self.n_pop]
        yis     = torch.zeros(yi.size()[0],2)
        yis[:,0]= 1-yi
        yis[:,1]= yi
        dV_dai  = dV_da[:,0]
        Vai     = Va_pred[:,0]
        ci      = torch.maximum(Vai, torch.tensor([1e-8],device=device))**(-1/self.gam)
        ys      = torch.matmul(torch.tensor(self.y_vals),
                    torch.transpose(yis, 0, 1)).to(device)
        si      = w*ys + r*ai - ci

        ## Find the value when y switched: V_switched = V(a_i,1-y_i,s_{-i}) 
        x_switch_y  = x.detach().clone()
        x_switch_y[:,self.n_pop] = 1-x[:,self.n_pop]
        V_switch_y  = model(x_switch_y,Z)
        lamis       = torch.matmul(torch.tensor(self.lamsd,dtype=yis.dtype),
                        torch.transpose(yis, 0, 1)).to(device)
        
        ## Add to loss function for differential equation
        loss_i  = (self.rho*V[:,0] - dV_dai*si - r*V[:,0]- lamis*(V_switch_y - V)[:,0])
        diffeq  += loss_i - self.Psi_a(ai)

        ## Terms at (a^j,y^j), j>1
        ## ------------------------------
        del dV_dai, ci, si, ys, x_switch_y
        for j in range(1,self.n_pop):
            ## Switch columns: 
            ## xj_switch = (a^j,...a^1,...,a^{n_pop},y^j,y^2...y^1,...)
            ## construct equilibrium from agent j's perspective and find sj.
            xj_switch   = x.detach().clone()
            ind         = (0,j,self.n_pop,self.n_pop+j)
            indx        = (j,0,self.n_pop+j,self.n_pop)
            xj_switch[:,ind] = x[:,indx].detach().clone()
            aj          = xj_switch[:,0]
            yj          = xj_switch[:,self.n_pop]
            yjs         = torch.zeros(yj.size()[0],2)
            yjs[:,0]    = 1-yj
            yjs[:,1]    = yj
            
            ## Construct equilibrium from j's perspective
            Kj, Lj, rj, wj = self.calc_eqm(xj_switch,z)

            ## Policies
            Vj          = model(xj_switch,z)[:,0]
            cj          = torch.maximum(Vj, 
                            torch.tensor([1e-8],device=device))**(-1/self.gam)
            ys          = torch.matmul(torch.tensor(self.y_vals),
                            torch.transpose(yjs, 0, 1)).to(device)
            sj          = wj*ys + rj*aj - cj

            ## Switch y
            ## Construct terms of V(a^1,...a^{n_pop},y^1,...,(1-y^j),...)
            x_switch_yj = torch.clone(x)
            x_switch_yj[:,self.n_pop+j] = torch.remainder(x[:,self.n_pop+j],-2)+1
            V_switch_yj = model(x_switch_yj,z)
            lamjs       = torch.matmul(torch.tensor(self.lamsd,dtype=yjs.dtype),
                            torch.transpose(yjs, 0, 1)).to(device)

            ## Add to differential equation
            dV_daj      = dV_da[:,j]
            loss_j      =- dV_daj*sj - lamjs*(V_switch_yj - V)[:,0]
            diffeq      += loss_j
        ## End tracking
        a.requires_grad_(False)
        z.requires_grad_(True)
        # z.requires_grad_(True)
        def model_z(z):
            return model(x, z)

        Vz_pred = model_z(z)
        dV_dz = self.get_derivs_1order(Vz_pred, z)
        dV_dz2 = self.get_derivs_1order(dV_dz, z)

        diffeq += -dV_dz*self.eta*(self.Zprod - Z) - 0.5 * self.sig**2 * dV_dz2

        ## End tracking
        z.requires_grad_(False)
        
        ## Return
        return diffeq

## --------------------------------------------------
## C. Training Sampler
## --------------------------------------------------
class Training_Sampler(Environments):
        
    def __init__(   self,
                    params,
                    dim,
                    sample_points_distribution,
                    batch_points_distribution,
                    locations):
        '''
        Initialize the class:
        dim         = dimension, in the problem, 1
        locations   = partition of [a_min, a_max] 
        sample_points_distribution  = distribution over locations
        batch_points_distribution   = distribution over locations in batchs
        '''
        super().__init__(params)
        self.dim = dim
        self.locations = locations
        self.sample_points_distribution = sample_points_distribution
        self.batch_points_distribution = batch_points_distribution
        self.sample_points_num = int(np.sum(sample_points_distribution))
        self.batch_points_num = int(np.sum(batch_points_distribution))
        self.num_intervals = locations.shape[0]
        lam1,lam2 = self.lams[0, 1],self.lams[1, 0]
        self.plam = [lam1/(lam1 + lam2), lam2/(lam1 + lam2)]
    def add_points(self,n,m):
        '''
        Add m points to location[i] (where the pde loss is the largest).
        See more in Lu et al. (2021).
        '''
        for j in range(n-2,n+3,1):
            if (j>=1) and (j<self.num_intervals):
                mp = int(m*2.0**(-abs(j-n)))
                self.sample_points_distribution[j] += mp
                self.batch_points_distribution[j] += mp
                self.sample_points_num += int(mp)
                self.batch_points_num += int(mp)

    def pretrain_sample(self,N,a_vals,g):
        '''
        Sampling in pretraining loop, by interpolation and stationary 
            distribution to replicate fd results
        N       = Number of points
        a_vals  = grid points on [a_min, a_max]
        g       = the solved stationary stationary distribution from fd method
        '''
        Z = (self.Zprod_max- self.Zprod_min)*np.random.rand(N) + self.Zprod_min
        # Use the stationary distribution to calibrate
        y       = np.random.choice(
                    np.arange(0, 2), (N, self.n_pop-1), 
                    p = self.plam
                    ).reshape((N, self.n_pop-1))
        a       = np.multiply(
                    np.random.choice(a_vals,(N,self.n_pop-1),
                    p = g[0,:]/np.sum(g[0,:])),1-y) \
                + np.multiply(
                    np.random.choice(a_vals,(N,self.n_pop-1),
                    p = g[1,:]/np.sum(g[1,:])),y)
        a_i     = self.a_min + (self.a_max-self.a_min)*lhs(1, N)
        k_oth   = (np.sum(a, axis = 1))/(self.n_pop-1)
        
        # Sample random interest rates between (r_lb, r_lb+r_rb] 
        # to scale other agent's asset holding
        r_s     = np.random.rand(N)*self.r_rb + self.r_lb
        k_s     = self.k_inv(r_s,Z)
        a_p_oth = a*k_s[:,None]/k_oth[:,None]
        a_s     = np.hstack((a_i, a_p_oth))
        yy      = np.random.choice(np.arange(0, 2), (N, self.n_pop), 
                    p = self.plam
                    ).reshape((N, self.n_pop))
        X       = np.hstack((a_s, yy))
        return X, Z.reshape((N, 1))
    
    def sample_al(self,N,coords):
        '''
        Sampling N points uniformly in the active learning region, where we 
        expect the loss is large, to imporve the learning process.
        '''
        # sample for N points in the interval mechanically, 
        # a_i around [r_lb, r_lb + r_rb]
        Z = (self.Zprod_max- self.Zprod_min)*np.random.rand(N) + self.Zprod_min

        a       = self.a_min + (self.a_max-self.a_min)*lhs(self.n_pop - 1, N)
        a_i     = coords[0,0] + (coords[1,0]-coords[0,0])*lhs(1, N)
        a       = np.random.rand(self.n_pop - 1) * a
        k_oth   = (np.sum(a, axis = 1))/(self.n_pop-1)
        
        # sample random interest rates between (r_lb, r_lb+r_rb]
        r_s     = np.random.rand(N)*self.r_rb + self.r_lb
        k_s     = self.k_inv(r_s,Z)
        a_p_oth = a*k_s[:,None]/k_oth[:,None]
        a_s     = np.hstack((a_i, a_p_oth))

        y       = np.random.choice(np.arange(0, 2), (N, self.n_pop),
                    p = self.plam
                    ).reshape((N, self.n_pop))
        X       = np.hstack((a_s, y))
        return X, Z.reshape((N, 1))
    
    def sample_main(self):
        '''
        Sampling points according to the distribution
        '''
        N       = self.sample_points_num
        Z = (params["Zprod_max"]- params["Zprod_min"])*np.random.rand(N) + params["Zprod_min"]
        a       = self.a_min + (self.a_max-params["a_min"])*lhs(params["n_pop"] - 1, N)
        a_i     = np.array([])
        for j in range(self.num_intervals):
            # for each small intervals, we sample the number of points at 
            # this location according to sample_points_distribution.
            n_j = int(self.sample_points_distribution[j])
            a_j = self.locations[j,0] + (self.locations[j,1]\
                 - self.locations[j,0]) * lhs(1,n_j)
            a_i = np.append(a_i,a_j)
        a_i     = np.reshape(a_i,(N,-1))
        a       = np.random.rand(self.n_pop - 1) * a
        k_oth   = (np.sum(a, axis = 1))/(self.n_pop-1)
        
        # Sample random interest rates between (r_lb, r_lb+r_rb]
        r_s     = np.random.rand(N)*self.r_rb + self.r_lb
        k_s     = self.k_inv(r_s,Z)
        a_p_oth = a*k_s[:,None]/k_oth[:,None]
        a_s     = np.hstack((a_i, a_p_oth))
        y       = np.random.choice(
                    np.arange(0, 2), (N, self.n_pop),
                    p = self.plam).reshape((N, self.n_pop))
        X       = np.hstack((a_s, y))
        return X, Z.reshape((N, 1))

    def sample_batch(self):
        '''
        batch sampling for mechanically al region and main sampling, 
        N_al al points, n_al al points (batched)
        '''
        a_id = np.array([])
        a_id_range = np.append(0,np.cumsum(self.sample_points_distribution))
        for j in range(self.num_intervals):
            n_j = int(self.batch_points_distribution[j])
            aj_id = np.random.choice(np.arange(a_id_range[j],a_id_range[j+1]),n_j)
            a_id = np.append(a_id,aj_id)
        return a_id.astype(int)
    def sample_batch_al(self,N_al,n_al):
        return np.random.choice(np.arange(N_al),n_al).astype(int)

## --------------------------------------------------
## D. Neural Network Class
## --------------------------------------------------
# nn without symmetry
class Master_eminn(torch.nn.Module):
    def __init__(self,nn_width,nn_num_layers,n_pop):
        super(Master_eminn, self).__init__()
        layers = [torch.nn.Linear(2*n_pop+1, nn_width),torch.nn.Tanh()]
        for i in range(1,nn_num_layers):
            layers.append(torch.nn.Linear(nn_width, nn_width))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(nn_width, 1))

        self.net = torch.nn.Sequential(*layers)
        for i in range (0,nn_num_layers):
            torch.nn.init.xavier_normal_(self.net[2*i].weight)

    def forward(self, X,Z):
        return self.net(torch.cat((X, Z), 1))
    
class Master_PINN(torch.nn.Module):
    def __init__(self,nn_width,nn_num_layers):
        super(Master_PINN, self).__init__()
        layers = [torch.nn.Linear(2*self.n_pop+1, nn_width),torch.nn.Tanh()]
        for i in range(1,nn_num_layers):
            layers.append(torch.nn.Linear(nn_width, nn_width))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(nn_width, 1))

        self.net = torch.nn.Sequential(*layers)
        for i in range (0,nn_num_layers):
            torch.nn.init.xavier_normal_(self.net[2*i].weight)

    def forward(self, X,Z):
        return self.net(torch.cat((X, Z), 1))
    