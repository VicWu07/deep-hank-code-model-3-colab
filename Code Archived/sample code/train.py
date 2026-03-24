"""
filename: train.py
@authors: Zhouzhou Gu, Jonathan Payne
This file contains sample code for solving the Krusell-Smith Master Equation.
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
Imports from the following files:
    para.py: has parameters, and preliminary functions
    aiy_fd_crra.py: has code for finite difference aiyagari solution.
    ks_nn_crra.py: has code for neural network solution.
    plot.py: has code for plotting.
"""

import torch
import numpy as np
import os
import time
from torch.distributions.exponential import Exponential
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.uniform import Uniform
from torch.distributions.categorical import Categorical
from datetime import datetime
import torch.optim as optim
from scipy.interpolate import RegularGridInterpolator
from torch.autograd import grad
## Load custom functions
from para import *                  # Load the parameters
from aiy_fd_crra import Household   # Load functions for finite difference
from ks_nn_crra import *           # Load neural net and sampler methods 

torch.manual_seed(777)

## Check if gpu is available

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## --------------------------------------------------
## --------------------------------------------------
## 1. Finite Difference Solution
## --------------------------------------------------
## Initialize the class
am = Household(params)

## Bisection loop to find the stationary equilibrium
am.bisection_loop()

## --------------------------------------------------
## --------------------------------------------------
## 2. Neural Network Solution
## --------------------------------------------------
# Create the directory according to current time
now         = datetime.now()
cur_time    = now.strftime("%Y_%m_%d_%H_%M_%S")
path        = '_'.join([cur_time,'popu',str(params["n_pop"])])
path        = os.path.join("Sample_outputs",path)
os.makedirs(path, exist_ok=True)

## --------------------------------------------------
## 2.A. Initialize neural net
## --------------------------------------------------
# Move the neural net parameters to "GPU", if available
eminn        = Master_eminn(params["nn_width"],
                            params["nn_num_layers"],
                            params["n_pop"]).to(device)
print(str(eminn))      
# Implementing data parallelism at module level in pytorch
eminn        = torch.nn.DataParallel(eminn) 
# Store model parameters as float variables instead of double to save space
eminn        = eminn.float()                                  
# Set up the optimizer as Adam, lr is the learning rate.
optimizer   = optim.Adam(eminn.parameters(), lr = 0.001);

## --------------------------------------------------
## 2.B. Initialize sampler
## --------------------------------------------------

# Partition [a_min, a_max] into Num_intervals of intervals evenly
locations = np.array([(x / params["Num_intervals"], (x + 1) 
                       / params["Num_intervals"]) \
                for x in range(params["Num_intervals"])])*(
                    params["a_max"]-params["a_min"]) + params["a_min"]

# Start from the uniform distributed training points
sample_points_dist = np.ones(params["Num_intervals"])*params["Num_sample_points"]
batch_points_dist = np.ones(params["Num_intervals"])*params["Num_batch_points"]
Sampler_main = Training_Sampler(params, 1, sample_points_dist, 
                    batch_points_dist, locations)
Sampler_main.add_points(14,16)

## --------------------------------------------------
## 2.C. Pretraining
## --------------------------------------------------
## Pretrains the neural network to match an initial function

## Define the initial y values, and interpolation
aa = am.a_vals[1:]
yy = np.array([0., 1.])
data = am.cf**(-params["gam"])
interp = RegularGridInterpolator((yy,aa)
        , data,bounds_error=False, fill_value=None)

def y_init(x,option):
    '''
    Pretrain the model, 3 possible options:
        1. option = "fd", Use fd solution to pretrain eminn
        Return the value at points between grids
        2. option = "log", Use -log(x) function to pretrain eminn
        3. option = "exp", Use exp(-x) function to pretrain eminn
    '''
    xx = torch.tensor(x,device=device)
    if option == 'fd':
        return torch.tensor(interp(x[:,[params["n_pop"],0]])
                            , dtype=torch.float32,device = device)
    elif option == 'log':
        return -torch.log(xx[:,0])
    elif option == 'exp':
        return torch.exp(-xx[:,0])

pretrain_option = "exp"
epochs = 3000

# Pre-Training Loop
# Train the model to match the value functions in stationary equilibrium
for epoch in range(1, epochs, 1):  # loop over the dataset multiple times
    X_pretrain,Z_pretrain = Sampler_main.pretrain_sample(100,am.a_vals,am.g)

    # transform into tensor variables
    X_pretrain_tensor = torch.tensor(X_pretrain\
                        , requires_grad=True, dtype=torch.float32)
    Z_pretrain_tensor = torch.tensor(Z_pretrain\
                         , requires_grad=True, dtype=torch.float32)
    # zero the parameter gradients
    optimizer.zero_grad()

    # run input through the eminn
    outputs = eminn(X_pretrain_tensor,Z_pretrain_tensor)

    ## Loss function
    y_init_vals = y_init(X_pretrain,pretrain_option).reshape(100,1).to(device)
    loss = torch.mean(torch.square(outputs - y_init_vals))

    # backward propagation
    loss.backward()
    # update model parameters
    optimizer.step()

    # turn tensor variable into array
    total_l  = loss.detach().cpu().numpy()

    # Print loss value in specific epochs
    if epoch%10 == 0:
        print("Iter %d: Total loss = %.4e" % (epoch, total_l))

    # Check for convergence
    if total_l < 1e-6:
        print('Converged at epoch %s with training loss %s' % (epoch, total_l))
        break

optimizer = optim.Adam(eminn.parameters(), lr = 0.0001)


## --------------------------------------------------
## 2.D. Main Training
## --------------------------------------------------
Training = Training_pde(params)
## Load the plotting functions
from plot import plot_comparison

# define the active learning region and boundary region
al_coords = np.array([[params["a_min"], 0.0],[params["a_max"]/10, 1.]])
bc_coords = np.array([[params["a_min"], 0.0],[params["a_min"], 1.]])

# Document model parameters
fo = open(os.path.join(path,'model_parameters.txt'), "a")
string = f"Number of agents: {params['n_pop']:0.1f}\n\
        range of r: {params['r_lb']:0.2f}, {params['r_rb']+params['r_lb']:0.2f}]\n\
        epochs = {epochs}\nwidth:{params['nn_width']:0.2f}"
string = string + f"\nkappa:{params['kappa']:0.1f}\n\
        a_lb:{params['a_lb']:0.1f}\nwidth:{params['nn_width']:.1f}\n\
        num_layers:{params['nn_num_layers']:.1f}"
string = string + "\n pretrain option:" + pretrain_option
fo.write(string+'\n'+str(eminn))
fo.flush()

## --------------------------------------------------
## 2.D.a Main Training (Adam Optimizer)
## --------------------------------------------------
## Training Loop
epochs = 6001
plot_comparison(0,eminn,am,path)
V_M = (10*(am.w_ss * am.y_vals[1] + am.r_ss * am.a_max))**(-am.gam)

tic = time.perf_counter()
flag = 1
for epoch in range(1, epochs, 1):  # loop over the dataset multiple times
    
    # Setup the residual sampler, defined on [a_min,a_max]
    X_res_sample, Z_res_sample = Sampler_main.sample_main()
    # Setup the active learning sampler, defined on the active learning region
    X_al_sample, Z_al_sample = Sampler_main.sample_al(params["num_of_al_points"],
                                                      al_coords)
    # Setup the boundary condition sampler, defined on the boundary
    X_bc_sample, Z_bc_sample = Sampler_main.sample_al(params["num_of_bc_points"],
                                                      bc_coords)
    
    total_l_batch = torch.zeros(params["n_batch"])
    bc_l_batch = torch.zeros(params["n_batch"])
    r_l_batch = torch.zeros(params["n_batch"])
    p_l_batch = torch.zeros(params["n_batch"])
    
    for batchi in range(params["n_batch"]):
        
        # do batch sampling
        Y = Sampler_main.sample_batch()
        Y_al = Sampler_main.sample_batch_al(params["num_of_al_points"]
                                            ,params["num_of_al_points_batch"])
        Y_bc = Sampler_main.sample_batch_al(params["num_of_bc_points"]
                                            ,params["num_of_bc_points_batch"])
        
        # convert numpy to tensor
        X_res_batch_tensor = torch.tensor(X_res_sample[Y,:]
                                        ,dtype=torch.float32,device = device)
        X_al_batch_tensor = torch.tensor(X_al_sample[Y_al,:]
                                        ,dtype=torch.float32,device = device)
        X_bc_batch_tensor = torch.tensor(X_bc_sample[Y_bc,:]
                                        ,dtype=torch.float32,device = device)
        X_res_cat_batch_tensor = torch.cat((X_res_batch_tensor
                                        ,X_al_batch_tensor,X_bc_batch_tensor),0)
        
        Z_res_batch_tensor = torch.tensor(Z_res_sample[Y,:]
                                        ,dtype=torch.float32,device = device)
        Z_al_batch_tensor = torch.tensor(Z_al_sample[Y_al,:]
                                        ,dtype=torch.float32,device = device)
        Z_bc_batch_tensor = torch.tensor(Z_bc_sample[Y_bc,:]
                                        ,dtype=torch.float32,device = device)
        Z_res_cat_batch_tensor = torch.cat((Z_res_batch_tensor
                                        ,Z_al_batch_tensor,Z_bc_batch_tensor),0)

        # residuals: the pde loss: (pde_oper(V)-0)^2
        residuals = Training.pde_oper(eminn, X_res_cat_batch_tensor,
                                      Z_res_cat_batch_tensor)
        loss_res = torch.mean(torch.square(residuals))

        # loss_pos: impose loss if ∂V/∂a < 0
        outputs = eminn(X_res_cat_batch_tensor,Z_res_cat_batch_tensor)
        loss_pos = torch.mean(torch.square(torch.min(
                    1/V_M*torch.ones(outputs.size(),device=device)-1/outputs,
                    torch.zeros(outputs.size(),device=device))))
        
        # dvdz: impose loss if ∂V/∂z < 0.
        dvdz = Training.dvdz_oper(eminn,X_res_cat_batch_tensor,Z_res_cat_batch_tensor)
        loss_dvdz = torch.square(torch.max(torch.mean(dvdz),torch.zeros(1,
                                                            device = device)))

        # Total loss
        loss = 10**2*loss_res + loss_pos + 10 *  loss_dvdz
        
        # Convert a tensor variable into float
        total_l_batch[batchi] = loss.item()
        bc_l_batch[batchi] = loss_dvdz.item()
        r_l_batch[batchi] = loss_res.item()
        p_l_batch[batchi] = loss_pos.item()

        # set grad to be zero, backward propogation, and update model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(eminn.parameters(), 1)
        optimizer.step()
    
    # calculate the mean, and print it out
    total_l  = float(torch.mean(total_l_batch))
    # bc_l  = float(torch.mean(bc_l_batch))
    r_l  = float(torch.mean(r_l_batch))
    p_l = float(torch.mean(p_l_batch))
    if epoch % 1 == 0:
        fo = open(os.path.join(path,'output_loss.txt'), "a")
        string = "Iter %d: Total loss = %.4e, Res. loss = %.4e\
                , dvdz loss = %.4e, POS loss = %.4e"\
                 % (epoch, total_l, r_l, loss_dvdz, p_l)
        fo.write( (string+'\n') )
        fo.flush()
        fo = open(os.path.join(path,'output_loss.csv'), "a")
        string = "%d,%.4e,%.4e,%.4e" % (epoch, total_l, r_l, p_l)
        fo.write( (string+'\n') )
        fo.flush()
    if (epoch % 10 == 0) and (epoch >2000):
        # after a couple of epochs, find where the loss is the largest
        # and add extra points to that region
        X_res_sample,Z_res_sample = Sampler_main.sample_main()
        X_res_tensor = torch.tensor(X_res_sample,dtype=torch.float32,
                                    device = device)
        Z_res_tensor = torch.tensor(Z_res_sample,dtype=torch.float32,
                                    device = device)
                                   
        r_l_locations = np.zeros(params["Num_intervals"])
        a_id_range = np.append(0,
                    np.cumsum(Sampler_main.sample_points_distribution))

        for j in range(params["Num_intervals"]):
            a_id = np.arange(a_id_range[j],a_id_range[j+1]).astype(int)
            # use the relative error instead, if not, 
            # the pde loss close to a_min is always the largest
            residuals = torch.div(Training.pde_oper(eminn,X_res_tensor[a_id,:],
                                                    Z_res_tensor[a_id,:])
                                ,eminn(X_res_tensor[a_id,:],Z_res_tensor[a_id,:]))
            loss_res = torch.mean(torch.square(residuals))
            r_l_locations[j] = loss_res.item()
        
        # adding additional point to where the relative error is the largest
        max_id = r_l_locations[3:15].argmax()
        Sampler_main.add_points(max_id+3,params["num_additional_points"])
        fo = open(os.path.join(path,'points_adjusted.txt'), "a")

        # document where to add in location_points.csv file
        string = "Iter %d: add %.1e points to = %d" \
                % (epoch,params["num_additional_points"], max_id)
        fo.write( (string+'\n') )
        fo.flush()
        fo = open(os.path.join(path,'location_points.csv'), "a")
        string = "%d,%.1e,%d" % (epoch, max_id,params["num_additional_points"])
        fo.write( (string+'\n') )
        fo.flush()
        print(r_l_locations)
    if epoch % 100 == 0:
        # plot the results every certain number of epoches
        toc = time.perf_counter()
        fo = open(os.path.join(path,'time_cost.txt'), "a")
        string = f"Every 200 epoch costs {toc - tic:0.4f} seconds"
        fo.write( (string+'\n') )
        fo.flush()
        plot_comparison(epoch,eminn,am,path)
        tic = time.perf_counter()
        torch.save(eminn, os.path.join(path,'eminn_'+str(epoch)+'_'
                                       +str(params["n_pop"])+'.pth'))
    if (r_l<5e-6) and (bc_l < 2e-7):
        flag = epoch
        break
