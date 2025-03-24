import numpy as np
import pinocchio as pin
from problems import create_humanoid_taichi_problem, create_kuka_problem, create_solo12_problem

from mim_robots.robot_loader import load_pinocchio_wrapper
import example_robot_data
import mim_solvers
import pathlib
import os

from plot_config import LABELS, COLORS, LINESTYLES

import time
# Solvers
SOLVERS = ['CSQP']
        #    'OSQP']
        #    'HPIPM_DENSE', 
        #    'HPIPM_OCP']

name = "Kuka"
# name = "solo12"
# name = "Taichi"

SAVE_PLOT = True


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

file_name = 'data/' + name + "_qp_convergence.npz"
# file_name = '/home/skleff/SQP_REBUTAL_BENCH/constrained/' + name + "_qp_convergence.npz"
print("Loading " + file_name)
npzfile = np.load(file_name)

if('CSQP' in SOLVERS):
    mean_csqp_iter = np.mean(npzfile['csqp_iter'], axis=0)
    mean_csqp_dist = np.median(npzfile['csqp_dist'], axis=0)
    q25_csqp_dist = np.quantile(npzfile['csqp_dist'], 0.25, axis=0)
    q75_csqp_dist = np.quantile(npzfile['csqp_dist'], 0.75, axis=0)
if('OSQP' in SOLVERS):
    mean_osqp_iter = np.mean(npzfile['osqp_iter'], axis=0)
    mean_osqp_dist = np.median(npzfile['osqp_dist'], axis=0)
    q25_osqp_dist = np.quantile(npzfile['osqp_dist'], 0.25, axis=0)
    q75_osqp_dist = np.quantile(npzfile['osqp_dist'], 0.75, axis=0)


fig0, ax0 = plt.subplots(1, 1, figsize=(19.2,10.8))
if('CSQP' in SOLVERS): 
    ax0.plot(mean_csqp_iter, mean_csqp_dist, color=COLORS['CSQP'], linestyle=LINESTYLES['CSQP'], linewidth=4, label=LABELS['CSQP'])
    ax0.fill_between(mean_csqp_iter, q75_csqp_dist, q25_csqp_dist, facecolor=COLORS['CSQP'], alpha=0.5)
if('OSQP' in SOLVERS):
    ax0.plot(mean_osqp_iter, mean_osqp_dist, color=COLORS['OSQP'], linestyle=LINESTYLES['OSQP'], linewidth=4, label=LABELS['OSQP'])
    ax0.fill_between(mean_osqp_iter, q75_osqp_dist, q25_osqp_dist, facecolor=COLORS['OSQP'], alpha=0.5)
    # ax0.fill_between(mean_csqp_time, mean_csqp_dist+std_csqp_dist, mean_csqp_dist-std_csqp_dist, facecolor='r', alpha=0.5)

# Set axis and stuff
ax0.set_xlabel('Number of iterations', fontsize=26)
ax0.set_ylabel(r'$\vert\vert x - x^{\star}\vert\vert$', fontsize=26)
ax0.set_yscale("log")
ax0.tick_params(axis = 'y', labelsize=22)
ax0.tick_params(axis = 'x', labelsize=22)
ax0.set_xlim(0, mean_csqp_iter[-1])
ax0.grid(True) 
# Legend 
handles0, labels0 = ax0.get_legend_handles_labels()
plt.legend(fontsize=26, loc='upper right')
if(SAVE_PLOT):
    fig0.savefig('figures/QP_convergence_analysis_'+name+'.pdf', bbox_inches="tight")
    fig0.savefig('/home/skleff/SQP_REBUTAL_BENCH/constrained/QP_convergence_analysis_'+name+'.pdf', bbox_inches="tight")
plt.show()