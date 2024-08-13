import numpy as np
import pinocchio as pin
from problems import create_humanoid_taichi_problem, create_kuka_problem, create_solo12_problem

from mim_robots.robot_loader import load_pinocchio_wrapper
import example_robot_data
import mim_solvers
import pathlib
import os
# python_path = pathlib.Path('/home/ajordana/eigen_workspace/mim_solvers/python/').absolute()
python_path = pathlib.Path('/home/skleff/workspace_native/mim_solvers/python/').absolute()
os.sys.path.insert(1, str(python_path))
from csqp import CSQP
from plot_config import LABELS, COLORS, LINESTYLES

import time
# Solvers
SOLVERS = ['CSQP',
        #    'OSQP',
        #    'HPIPM_DENSE', 
           'HPIPM_OCP']

TOL         = 0.
CALLBACKS   = False
EPS_IP       = 1e-8
EPS_REL     = 0.


# name = "solo12"
name = "kuka"
# name = "taichi"



if name == "solo12":
    MAX_QP_ITER_HPIPM = 10 
    MAX_QP_ITER_ADMM  = 100


if name == "taichi":
    MAX_QP_ITER_HPIPM = 50
    MAX_QP_ITER_ADMM  = 250

if name == "kuka":
    MAX_QP_ITER_HPIPM = 10 
    MAX_QP_ITER_ADMM  = 200 
    kuka = load_pinocchio_wrapper("iiwa")

SEED = 10 ; np.random.seed(SEED)
n_samples = 5

def distance(x, y):
    return np.linalg.norm(x - y, np.inf)

csqp_time = []
csqp_dist = []

hpipm_ocp_time = []
hpipm_ocp_dist = []


# n_samples = 1000

for sample_id in range(n_samples):
    print("\n Sample ID = ", sample_id)

    if(name == "kuka"):
        q0 = np.array([0., 1.04, 0., -1.13, 0.2,  0.78, 0])     # Initial robot joint configuration
        kuka_x0 = np.concatenate([q0 + 2 * (np.random.random(kuka.model.nv) - 0.5), 2 * (np.random.random(kuka.model.nv) - 0.5)])
        pb = create_kuka_problem(kuka_x0) 

    if(name == "solo12"):
        mu = 0.8 + 0.01 * (2*np.random.rand() - 1)
        pb = create_solo12_problem(mu)

    if(name == "taichi"):
        err = np.zeros(3)
        err[2] = 2*np.random.rand() - 1
        taichi_p0    = np.array([0.4, 0, 1.2])
        p0 = taichi_p0 + 0.05*err
        pb = create_humanoid_taichi_problem(p0)


    # Create solver CSQP 
    if('CSQP' in SOLVERS):
        solvercsqp = mim_solvers.SolverCSQP(pb)
        solvercsqp.termination_tolerance = TOL
        solvercsqp.with_qp_callbacks = False
        solvercsqp.eps_abs = 0.
        solvercsqp.eps_rel = EPS_REL
        solvercsqp.equality_qp_initial_guess = False
        solvercsqp.update_rho_with_heuristic = False
        solvercsqp.with_callbacks = CALLBACKS
        # solvercsqp.with_qp_callbacks = True # CALLBACKS

    # Create solver HPIPM ocp
    if('HPIPM_OCP' in SOLVERS):
        solverhpipm_ocp = CSQP(pb, "HPIPM_OCP")
        solverhpipm_ocp.termination_tolerance  = TOL
        solverhpipm_ocp.eps_abs = EPS_IP
        solverhpipm_ocp.eps_rel = EPS_REL
        solverhpipm_ocp.with_callbacks = CALLBACKS





    # use GT hpipm 
    solverGT = CSQP(pb, "HPIPM_OCP")
    solverGT.termination_tolerance  = TOL
    solverGT.xs = [solverGT.problem.x0] * (solverGT.problem.T + 1) 
    solverGT.us = solverGT.problem.quasiStatic([solverGT.problem.x0] * solverGT.problem.T)
    solverGT.max_qp_iters = 10000
    solverGT.eps_abs = EPS_IP
    solverGT.eps_rel = 0.
    solverGT.with_callbacks = False
    solverGT.solve(solverGT.xs, solverGT.us, 1, False)
    solved = (solverGT.norm_primal < EPS_IP and solverGT.norm_dual < EPS_IP)
    assert solved
    print("qp_iter HPIPM GT = ", solverGT.qp_iters)

    # # use GT csqp to obtain ground truth 
    # solverGT_CSQP = mim_solvers.SolverCSQP(pb)
    # solverGT_CSQP.termination_tolerance  = TOL
    # solverGT_CSQP.xs = [solverGT_CSQP.problem.x0] * (solverGT_CSQP.problem.T + 1) 
    # solverGT_CSQP.us = solverGT_CSQP.problem.quasiStatic([solverGT_CSQP.problem.x0] * solverGT_CSQP.problem.T)
    # solverGT_CSQP.max_qp_iters = 10000
    # solverGT_CSQP.eps_abs = EPS_IP
    # solverGT_CSQP.eps_rel = 0.
    # solverGT_CSQP.with_callbacks = False
    # solverGT_CSQP.solve(solverGT_CSQP.xs, solverGT_CSQP.us, 0, False)
    # solverGT_CSQP.computeDirection(True)
    # solved = (solverGT_CSQP.norm_primal < EPS_IP and solverGT_CSQP.norm_dual < EPS_IP)
    # assert solved
    # print("qp_iter CSQP GT = ", solverGT_CSQP.qp_iters)


    dx_GT = np.array(solverGT.dx, copy=True)
    du_GT = np.array(solverGT.du, copy=True)

    # dx_GT_CSQP = np.array(solverGT_CSQP.dx, copy=True)
    # du_GT_CSQP = np.array(solverGT_CSQP.du, copy=True)


    dx = np.zeros_like(dx_GT)
    du = np.zeros_like(du_GT)
    dist_HPIPM = distance(dx, dx_GT) + distance(du, du_GT)

    dx = np.zeros_like(dx_GT)
    du = np.zeros_like(du_GT)
    # dist_CSQP = distance(dx, dx_GT_CSQP) + distance(du, du_GT_CSQP)
    dist_CSQP = distance(dx, dx_GT) + distance(du, du_GT)

    sample_csqp_time = [0.]
    sample_csqp_dist = [dist_CSQP]

    sample_hpipm_ocp_time = [0.]
    sample_hpipm_ocp_dist = [dist_HPIPM]
    for iter in range(1, MAX_QP_ITER_ADMM+1, 10):
        # CSQP
        if('CSQP' in SOLVERS):
            # print("   Problem : CSQP")
            solvercsqp.max_qp_iters = iter
            # solvercsqp.rho_update_interval = 20 # 10 current best 
            solvercsqp.xs = [solvercsqp.problem.x0] * (solvercsqp.problem.T + 1) 
            solvercsqp.us = solvercsqp.problem.quasiStatic([solvercsqp.problem.x0] * solvercsqp.problem.T)
            solvercsqp.solve(solvercsqp.xs, solvercsqp.us, 0, False)
            # import pdb; pdb.set_trace()
            t1 = time.time()
            solvercsqp.computeDirection(True)
            solvercsqp.qp_time = time.time() - t1
            dx = np.array(solvercsqp.dx)
            du = np.array(solvercsqp.du)
            # dist = np.max((solvercsqp.norm_primal, solvercsqp.norm_dual)) # 
            dist = distance(dx, dx_GT) + distance(du, du_GT)
            sample_csqp_time.append(solvercsqp.qp_time*1e3)
            sample_csqp_dist.append(dist)
            # print("csqp iters = ", solvercsqp.qp_iters)
            # print("csqp norm = ", np.max((solvercsqp.norm_primal, solvercsqp.norm_dual)))


    for iter in range(1, MAX_QP_ITER_HPIPM+1):
        print(iter, " / ", MAX_QP_ITER_HPIPM)
        # HPIPM_OCP    
        if('HPIPM_OCP' in SOLVERS):    
            print("   Problem :  HPIPM_OCP")
            solverhpipm_ocp.max_qp_iters = iter
            solverhpipm_ocp.xs = [solverhpipm_ocp.problem.x0] * (solverhpipm_ocp.problem.T + 1) 
            solverhpipm_ocp.us = solverhpipm_ocp.problem.quasiStatic([solverhpipm_ocp.problem.x0] * solverhpipm_ocp.problem.T)
            solverhpipm_ocp.solve(solverhpipm_ocp.xs, solverhpipm_ocp.us, 1, False)
            dx = np.array(solverhpipm_ocp.dx)
            du = np.array(solverhpipm_ocp.du)
            dist = distance(dx, dx_GT) + distance(du, du_GT)
            # dist = np.max((solverhpipm_ocp.norm_primal, solverhpipm_ocp.norm_dual)) 
            sample_hpipm_ocp_time.append(solverhpipm_ocp.qp_time*1e3)
            sample_hpipm_ocp_dist.append(dist)
            print("hpipm iters = ", solverhpipm_ocp.qp_iters)
            # print("hpipm norm = ", np.max((solverhpipm_ocp.norm_primal, solverhpipm_ocp.norm_dual)))

    csqp_time.append(sample_csqp_time)
    csqp_dist.append(sample_csqp_dist)

    hpipm_ocp_time.append(sample_hpipm_ocp_time)
    hpipm_ocp_dist.append(sample_hpipm_ocp_dist)

# convert to array
csqp_time = np.array(csqp_time)
csqp_dist = np.array(csqp_dist)

hpipm_ocp_time = np.array(hpipm_ocp_time)
hpipm_ocp_dist = np.array(hpipm_ocp_dist)

# Extract mean and std
mean_csqp_time = np.mean(csqp_time, axis=0)
mean_hpipm_ocp_time = np.mean(hpipm_ocp_time, axis=0)

mean_csqp_dist = np.median(csqp_dist, axis=0)
mean_hpipm_ocp_dist = np.median(hpipm_ocp_dist, axis=0)

q25_csqp_dist = np.quantile(csqp_dist, 0.25, axis=0)
q25_hpipm_ocp_dist = np.quantile(hpipm_ocp_dist, 0.25, axis=0)

q75_csqp_dist = np.quantile(csqp_dist, 0.75, axis=0)
q75_hpipm_ocp_dist = np.quantile(hpipm_ocp_dist, 0.75, axis=0)

std_csqp_dist = np.std(csqp_dist, axis=0)
std_hpipm_ocp_dist = np.std(hpipm_ocp_dist, axis=0)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
 
fig0, ax0 = plt.subplots(1, 1, figsize=(19.2,10.8))
if('CSQP' in SOLVERS): 
    ax0.plot(mean_csqp_time, mean_csqp_dist, color=COLORS['CSQP'], linestyle=LINESTYLES['CSQP'], linewidth=4, label=LABELS['CSQP'])
    ax0.fill_between(mean_csqp_time, q75_csqp_dist, q25_csqp_dist, facecolor=COLORS['CSQP'], alpha=0.5)
    # ax0.fill_between(mean_csqp_time, mean_csqp_dist+std_csqp_dist, mean_csqp_dist-std_csqp_dist, facecolor='r', alpha=0.5)

if('HPIPM_OCP' in SOLVERS): 
    ax0.plot(mean_hpipm_ocp_time, mean_hpipm_ocp_dist, color=COLORS['HPIPM_OCP'], linestyle=LINESTYLES['HPIPM_OCP'], linewidth=4, label=LABELS['HPIPM_OCP'])
    ax0.fill_between(mean_hpipm_ocp_time, q75_hpipm_ocp_dist, q25_hpipm_ocp_dist, facecolor=COLORS['HPIPM_OCP'], alpha=0.5)
    # ax0.fill_between(mean_hpipm_ocp_time, mean_hpipm_ocp_dist+std_hpipm_ocp_dist, mean_hpipm_ocp_dist-std_hpipm_ocp_dist, facecolor='b', alpha=0.5)

# Set axis and stuff
ax0.set_xlabel('Time [ms]', fontsize=26)
ax0.set_ylabel(r'$\vert\vert x - x^{\star}\vert\vert$', fontsize=26)
ax0.set_yscale("log")
ax0.tick_params(axis = 'y', labelsize=22)
ax0.tick_params(axis = 'x', labelsize=22)
ax0.set_xlim(0, max(mean_csqp_time[-1], mean_hpipm_ocp_time[-1]))
ax0.grid(True) 
# Legend 
handles0, labels0 = ax0.get_legend_handles_labels()
fig0.legend(handles0, labels0, loc='lower right', prop={'size': 26}) 
fig0.savefig('/tmp/convergence_plots_'+name+'.pdf', bbox_inches="tight")
plt.show()