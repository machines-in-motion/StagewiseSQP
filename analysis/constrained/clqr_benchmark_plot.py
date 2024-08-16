from clqr import ActionModelCLQR
import mim_solvers
import numpy as np
import time
import crocoddyl
import pathlib
import os
from plot_config import LABELS, COLORS, LINESTYLES, LABELSIZE, FONTSIZE, FIGSIZE


# Generate plot of number of iterations for each problem
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

SAVE = True
# Solvers
SOLVERS = ['CSQP',
           'OSQP']
        #    'HPIPM_DENSE', 
        #    'HPIPM_OCP']

WHICH_PLOT = ['horizon', 'state'] 

PREFIX = 'data/'
if('horizon' in WHICH_PLOT) :
    # Load file
    horizon_file = PREFIX + "CLQR_horizon_benchmark.npz"
    print("Loading " + horizon_file)
    npz_horizon = np.load(horizon_file)
    # x-axis : max time allowed to solve the QP (in ms)
    xdata     = np.array(npz_horizon['dim_list'])
    print("horizon_list : ", npz_horizon['dim_list'])
    print("nx : ", npz_horizon['nx'])
    print("N_samples : ", npz_horizon['N_samples'])
    fig0 = plt.figure(figsize=FIGSIZE)
    if('CSQP' in SOLVERS):
        plt.plot(xdata, npz_horizon['csqp_qp_time_mean'], color=COLORS['CSQP'], linestyle=LINESTYLES['CSQP'], linewidth=4, label=LABELS['CSQP']) 
        plt.fill_between(xdata, npz_horizon['csqp_qp_time_mean']+npz_horizon['csqp_qp_time_std'], npz_horizon['csqp_qp_time_mean']-npz_horizon['csqp_qp_time_std'], facecolor=COLORS['CSQP'], alpha=0.5)
    if('OSQP' in SOLVERS):
        plt.plot(xdata, npz_horizon['osqp_qp_time_mean'], color=COLORS['OSQP'], linestyle=LINESTYLES['OSQP'], linewidth=4, label=LABELS['OSQP'])
        plt.fill_between(xdata, npz_horizon['osqp_qp_time_mean']+npz_horizon['osqp_qp_time_std'], npz_horizon['osqp_qp_time_mean']-npz_horizon['osqp_qp_time_std'], facecolor=COLORS['OSQP'], alpha=0.5)
    if('HPIPM_DENSE' in SOLVERS):
        plt.plot(xdata, npz_horizon['hpipm_dense_qp_time_mean'], color=COLORS['HPIPM_DENSE'], linestyle=LINESTYLES['HPIPM_DENSE'], linewidth=4, label=LABELS['HPIPM_DENSE'])
        plt.fill_between(xdata, npz_horizon['hpipm_dense_qp_time_mean']+npz_horizon['hpipm_dense_qp_time_std'], npz_horizon['hpipm_dense_qp_time_mean']-npz_horizon['hpipm_dense_qp_time_std'], facecolor=COLORS['HPIPM_DENSE'], alpha=0.5)
    if('HPIPM_OCP' in SOLVERS):
        plt.plot(xdata, npz_horizon['hpipm_ocp_qp_time_mean'], color=COLORS['HPIPM_OCP'], linestyle=LINESTYLES['HPIPM_OCP'], linewidth=4, label=LABELS['HPIPM_OCP'])
        plt.fill_between(xdata, npz_horizon['hpipm_ocp_qp_time_mean']+npz_horizon['hpipm_ocp_qp_time_std'], npz_horizon['hpipm_ocp_qp_time_mean']-npz_horizon['hpipm_ocp_qp_time_std'], facecolor=COLORS['HPIPM_OCP'], alpha=0.5)
    # Set axis and stuff
    plt.ylabel('Time [ms]', fontsize=FONTSIZE)
    plt.xlabel('Horizon length', fontsize=FONTSIZE)
    plt.tick_params(axis = 'y', labelsize=LABELSIZE)
    plt.tick_params(axis = 'x', labelsize=LABELSIZE)
    plt.xticks(npz_horizon['dim_list'])
    plt.grid(True) 
    # Legend 
    plt.legend(loc='upper left', prop={'size': FONTSIZE}) 
    # Save, show , clean
    if(SAVE):
        fig0.savefig('/tmp/CLQR_benchmark_horizon.pdf', bbox_inches="tight")


if('state' in WHICH_PLOT): 
    # Load file
    state_file = PREFIX + "CLQR_state_benchmark.npz"
    print("Loading " + state_file)
    npz_state = np.load(state_file)
    xdata     = np.array(npz_state['dim_list'])
    print("state_dim_list : ", npz_state['dim_list'])
    print("horizon : ", npz_state['horizon'])
    print("N_samples : ", npz_state['N_samples'])
    fig0 = plt.figure(figsize=FIGSIZE)
    if('CSQP' in SOLVERS):
        plt.plot(xdata, npz_state['csqp_qp_time_mean'], color=COLORS['CSQP'], linestyle=LINESTYLES['CSQP'], linewidth=4, label=LABELS['CSQP']) 
        plt.fill_between(xdata, npz_state['csqp_qp_time_mean']+npz_state['csqp_qp_time_std'], npz_state['csqp_qp_time_mean']-npz_state['csqp_qp_time_std'], facecolor=COLORS['CSQP'], alpha=0.5)
    if('OSQP' in SOLVERS):
        plt.plot(xdata, npz_state['osqp_qp_time_mean'], color=COLORS['OSQP'], linestyle=LINESTYLES['OSQP'], linewidth=4, label=LABELS['OSQP'])
        plt.fill_between(xdata, npz_state['osqp_qp_time_mean']+npz_state['osqp_qp_time_std'], npz_state['osqp_qp_time_mean']-npz_state['osqp_qp_time_std'], facecolor=COLORS['OSQP'], alpha=0.5)
    if('HPIPM_DENSE' in SOLVERS):
        plt.plot(xdata, npz_state['hpipm_dense_qp_time_mean'], color=COLORS['HPIPM_DENSE'], linestyle=LINESTYLES['HPIPM_DENSE'], linewidth=4, label=LABELS['HPIPM_DENSE'])
        plt.fill_between(xdata, npz_state['hpipm_dense_qp_time_mean']+npz_state['hpipm_dense_qp_time_std'], npz_state['hpipm_dense_qp_time_mean']-npz_state['hpipm_dense_qp_time_std'], facecolor=COLORS['HPIPM_DENSE'], alpha=0.5)
    if('HPIPM_OCP' in SOLVERS):
        plt.plot(xdata, npz_state['hpipm_ocp_qp_time_mean'], color=COLORS['HPIPM_OCP'], linestyle=LINESTYLES['HPIPM_OCP'], linewidth=4, label=LABELS['HPIPM_OCP'])
        plt.fill_between(xdata, npz_state['hpipm_ocp_qp_time_mean']+npz_state['hpipm_ocp_qp_time_std'], npz_state['hpipm_ocp_qp_time_mean']-npz_state['hpipm_ocp_qp_time_std'], facecolor=COLORS['HPIPM_OCP'], alpha=0.5)
    # Set axis and stuff
    plt.ylabel('Time [ms]', fontsize=FONTSIZE)
    plt.xlabel('State dimension', fontsize=FONTSIZE)
    plt.tick_params(axis = 'y', labelsize=LABELSIZE)
    plt.tick_params(axis = 'x', labelsize=LABELSIZE)
    plt.xticks(npz_state['dim_list'])
    plt.grid(True) 
    # Legend 
    plt.legend(loc='upper left', prop={'size': FONTSIZE}) 
    # Save, show , clean
    if(SAVE):
        fig0.savefig('/tmp/CLQR_benchmark_state.pdf', bbox_inches="tight")

plt.show()
plt.close('all')