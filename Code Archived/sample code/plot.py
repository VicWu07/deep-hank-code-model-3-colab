"""
filename: plot.py (adapted from aiyagari_penalty_func_12_28.py)
@authors: Zhouzhou Gu, Jonathan Payne

This file plots the comparison for consumption c(x), solution v'(x) by neural nets with the finite difference's solution

"""

import seaborn as sb
import os
import matplotlib.pyplot as plt
from para import *
from torch.distributions.categorical import Categorical

def plot_comparison(iteration_num,eminn,am,path):
    """ Plot the c(x), V'(x) for the model, and save plots to given path
        iteration_num   which epoch
        eminn            the neural net model
        am              fd solution
        path            the directory created by current time
    """
    N_p = am.N_FD
    D_in = 2*am.n_pop
    n_ido = 2
    Zss = torch.ones((N_p, 1))*am.Zprod
    a_p_vals = torch.linspace(am.a_min,am.a_max,N_p)

    colors = sb.color_palette()

    k_ss = am.k_inv(am.r_ss)
    a_ss = torch.tensor(am.a_vals)
    g_ss = torch.tensor(np.maximum(am.g,0.0*am.g))
    n_ido = 2

    m_ss_1 = Categorical(g_ss[0,:])
    m_ss_2 = Categorical(g_ss[1,:])

    a_p_oth_1_id = m_ss_1.sample(sample_shape=torch.Size([int(params["n_pop"]/2)]))
    a_p_oth_2_id = m_ss_2.sample(sample_shape=torch.Size([int(params["n_pop"]/2)]))
    a_p_oth_1_raw = a_ss[a_p_oth_1_id]
    a_p_oth_2_raw = a_ss[a_p_oth_2_id]


    k_oth = (sum(a_p_oth_1_raw)+sum(a_p_oth_2_raw))/(params["n_pop"]-1)
    a_p_oth_1 = a_p_oth_1_raw*k_ss/k_oth
    a_p_oth_2 = a_p_oth_2_raw*k_ss/k_oth

    x_p_y1 = torch.zeros(N_p, 2*params["n_pop"])
    x_p_y1[:,0] = a_p_vals
    x_p_y1[:,1:int(params["n_pop"]/2)+1] = a_p_oth_1
    x_p_y1[:,int(params["n_pop"]/2)+1:params["n_pop"]] = a_p_oth_2
    x_p_y1[:,params["n_pop"]] = 0
    x_p_y1[:,params["n_pop"]+1:n_ido*params["n_pop"]-int(params["n_pop"]/2)] = 0
    x_p_y1[:,n_ido*params["n_pop"]-int(params["n_pop"]/2):n_ido*params["n_pop"]] = 1

    x_p_y2 = torch.zeros(N_p, 2*params["n_pop"])
    x_p_y2[:,0] = a_p_vals
    x_p_y2[:,1:int(params["n_pop"]/2)+1] = a_p_oth_1
    x_p_y2[:,int(params["n_pop"]/2)+1:params["n_pop"]] = a_p_oth_2
    x_p_y2[:,params["n_pop"]] = 1
    x_p_y2[:,params["n_pop"]+1:n_ido*params["n_pop"]-int(params["n_pop"]/2)] = 0
    x_p_y2[:,n_ido*params["n_pop"]-int(params["n_pop"]/2):n_ido*params["n_pop"]] = 1

    y_pred_1 = eminn(x_p_y1,Zss)
    y_pred_2 = eminn(x_p_y2,Zss)
    c_1         = (y_pred_1)**(-1/params["gam"])
    c_2         = (y_pred_2)**(-1/params["gam"])
    x_p_vals_np = a_p_vals.detach().cpu().numpy().reshape(-1)
    c_1_np = c_1.detach().cpu().numpy().reshape(-1)
    c_2_np = c_2.detach().cpu().numpy().reshape(-1)
    dV_1_np = y_pred_1.detach().cpu().numpy().reshape(-1)
    dV_2_np = y_pred_2.detach().cpu().numpy().reshape(-1)

    ## Plot figures: consumption rule
    fig, ax = plt.subplots(figsize=(8,5))
    sb.set(style="whitegrid")
    ax.plot(x_p_vals_np[0:N_p], c_1_np[0:N_p],
        label=r'$c^{NN}(a,y_1,s^{SS})$',
        linestyle='-', color = colors[0])
    ax.plot(x_p_vals_np[0:N_p], c_2_np[0:N_p],
        label=r'$c^{NN}(a,y_2,s^{SS})$',
        linestyle='-', color = colors[1])
    ax.plot(am.a_vals[0:(int(params["N_FD"]*(am.a_max/params["a_max_FD"])))],
        am.cf[0,0:int(params["N_FD"]*(am.a_max/params["a_max_FD"]))],
        lw=2, alpha=0.6, label=r'$c^{FD}(a,y_1,s^{SS})$',
        linestyle='--', color = colors[0])
    ax.plot(am.a_vals[0:(int(params["N_FD"]*(am.a_max/params["a_max_FD"])))],
        am.cf[1,0:int(params["N_FD"]*(am.a_max/params["a_max_FD"]))],
        lw=2, alpha=0.6, label=r'$c^{FD}(a,y_2,s^{SS})$',
        linestyle='--', color = colors[1])
    plt.xlabel('Assets (a)')
    plt.ylabel('Consumption (c)')
    sb.despine(left=True)
    ax.legend()

    plt.title(str(iteration_num))
    filename = 'c_comp_'+str(iteration_num)+'_.png'
    savepath = os.path.join(path, filename)
    plt.savefig(savepath)
    ## Plot figures: dV
    fig, ax = plt.subplots(figsize=(8,5))
    sb.set(style="whitegrid")
    ax.plot(x_p_vals_np[0:N_p], dV_1_np[0:N_p],
        label=r'$dV^{NN}(a,y_1,s^{SS})$',
        linestyle='-', color = colors[0])
    ax.plot(x_p_vals_np[0:N_p], dV_2_np[0:N_p],
        label=r'$dV^{NN}(a,y_2,s^{SS})$',
        linestyle='-', color = colors[1])
    ax.plot(am.a_vals[0:(int(params["N_FD"]*am.a_max/params["a_max_FD"]))],
        (am.cf[0,0:int(params["N_FD"]*am.a_max/params["a_max_FD"])])**(-params["gam"]),
        lw=2, alpha=0.6, label=r'$dV^{FD}(a,y_1,s^{SS})$',
        linestyle='--', color = colors[0])
    ax.plot(am.a_vals[0:(int(params["N_FD"]*am.a_max/params["a_max_FD"]))],
        (am.cf[1,0:int(params["N_FD"]*am.a_max/params["a_max_FD"])])**(-params["gam"]),
        lw=2, alpha=0.6, label=r'$dV^{FD}(a,y_2,s^{SS})$',
        linestyle='--', color = colors[1])
    plt.xlabel('Assets (a)')
    plt.ylabel('dV')
    sb.despine(left=True)
    ax.legend()
    plt.title(str(iteration_num))
    filename = 'dV_comp_'+str(iteration_num)+'_.png'
    savepath = os.path.join(path, filename)
    plt.savefig(savepath)
    plt.close('all')