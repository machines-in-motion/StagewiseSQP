import numpy as np
import pinocchio as pin
from problems import create_humanoid_taichi_problem, create_kuka_problem, create_solo12_problem

from mim_robots.robot_loader import load_pinocchio_wrapper
import example_robot_data
import mim_solvers
import pathlib
import os
python_path = pathlib.Path('/home/ajordana/eigen_workspace/mim_solvers/python/').absolute()
os.sys.path.insert(1, str(python_path))
from csqp import CSQP

import time
# Solvers
SOLVERS = ['CSQP',
        #    'OSQP',
        #    'HPIPM_DENSE', 
           'HPIPM_OCP']

TOL         = 0.
CALLBACKS   = False
EPS_ABS_ADMM     = 1e-100
EPS_IP           = 1e-10 # HPIPM can behave weirdly with an artificially low tolerance
EPS_REL     = 0.


if('CSQP' in SOLVERS):
    solversCSQP         = [] 
    csqp_iter_samples   = []  
    csqp_time_samples   = []
    csqp_solved_samples = []
if('OSQP' in SOLVERS):
    solversOSQP         = [] 
    osqp_iter_samples   = []  
    osqp_time_samples   = []
    osqp_solved_samples = []
if('HPIPM_DENSE' in SOLVERS):
    solversHPIPM_dense         = [] 
    hpipm_dense_iter_samples   = []  
    hpipm_dense_time_samples   = []
    hpipm_dense_solved_samples = []
if('HPIPM_OCP' in SOLVERS):
    solversHPIPM_ocp         = [] 
    hpipm_ocp_iter_samples   = []  
    hpipm_ocp_time_samples   = []
    hpipm_ocp_solved_samples = []

# name = "kuka"
# name = "solo12"
name = "taichi"


if name == "taichi":
    pb = create_humanoid_taichi_problem()

if name == "kuka": 
    q0 = np.array([0., 1.04, 0., -1.13, 0.2,  0.78, 0])
    x0 = np.concatenate([q0, np.zeros(7)]).copy()
    pb = create_kuka_problem(x0)

if name == "solo12": 
    pb = create_solo12_problem(0.8)

# use ProxQP to obtain ground truth 
solverGT = CSQP(pb, "ProxQP")
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
print("qp_iter = ", solverGT.qp_iters)


dx_GT = np.array(solverGT.dx, copy=True)
du_GT = np.array(solverGT.du, copy=True)

MAX_QP_ITER_HPIPM = 40
MAX_QP_ITER_ADMM  = 300


# Create solver CSQP 
if('CSQP' in SOLVERS):
    solvercsqp = mim_solvers.SolverCSQP(pb)
    solvercsqp.termination_tolerance = TOL
    solvercsqp.with_qp_callbacks = False
    solvercsqp.eps_abs = EPS_ABS_ADMM
    solvercsqp.eps_rel = EPS_REL
    solvercsqp.equality_qp_initial_guess = False
    solvercsqp.update_rho_with_heuristic = False
    solvercsqp.with_callbacks = CALLBACKS
    # solvercsqp.with_qp_callbacks = True # CALLBACKS

# Create solver OSQP
if('OSQP' in SOLVERS):
    solverosqp = CSQP(pb, "OSQP")
    solverosqp.termination_tolerance = TOL
    solverosqp.eps_abs = EPS_ABS_ADMM
    solverosqp.eps_rel = EPS_REL
    solverosqp.with_callbacks = CALLBACKS
    solverosqp.verboseQP = True

# Create solver HPIPM dense
if('HPIPM_DENSE' in SOLVERS):
    solverhpipm_dense = CSQP(pb, "HPIPM_DENSE")
    solverhpipm_dense.termination_tolerance  = TOL
    solverhpipm_dense.eps_abs = EPS_IP
    solverhpipm_dense.eps_rel = EPS_REL
    solverhpipm_dense.with_callbacks = CALLBACKS

# Create solver HPIPM ocp
if('HPIPM_OCP' in SOLVERS):
    solverhpipm_ocp = CSQP(pb, "HPIPM_OCP")
    solverhpipm_ocp.termination_tolerance  = TOL
    solverhpipm_ocp.eps_abs = EPS_IP
    solverhpipm_ocp.eps_rel = EPS_REL
    solverhpipm_ocp.with_callbacks = CALLBACKS


csqp_time = []
csqp_dx = []
csqp_du = []

osqp_time = []
osqp_dx = []
osqp_du = []

hpipm_ocp_time = []
hpipm_ocp_dx = []
hpipm_ocp_du = []

hpipm_time = []
hpipm_dx = []
hpipm_du = []


for iter in range(1, MAX_QP_ITER_ADMM+1):
    if iter%10 == 0:
        print(iter, " / ", MAX_QP_ITER_ADMM)
        
    # CSQP
    if('CSQP' in SOLVERS):
        # print("   Problem : CSQP")
        solvercsqp.max_qp_iters = iter
        solvercsqp.xs = [solvercsqp.problem.x0] * (solvercsqp.problem.T + 1) 
        solvercsqp.us = solvercsqp.problem.quasiStatic([solvercsqp.problem.x0] * solvercsqp.problem.T)
        solvercsqp.solve(solvercsqp.xs, solvercsqp.us, 0, False)
        t1 = time.time()
        solvercsqp.computeDirection(True)
        solvercsqp.qp_time = time.time() - t1
        print(solvercsqp.qp_iters)
        csqp_time.append(solvercsqp.qp_time*1e3)
        csqp_dx.append(np.array(solvercsqp.dx_tilde))
        csqp_du.append(np.array(solvercsqp.du_tilde))

for iter in range(1, MAX_QP_ITER_ADMM * 10 +1, 100):
    # OSQP
    if('OSQP' in SOLVERS):
        print("   Problem : OSQP")
        solverosqp.max_qp_iters = iter
        solverosqp.xs = [solverosqp.problem.x0] * (solverosqp.problem.T + 1) 
        solverosqp.us = solverosqp.problem.quasiStatic([solverosqp.problem.x0] * solverosqp.problem.T)
        solverosqp.solve(solverosqp.xs, solverosqp.us, 1, False)        
        osqp_time.append(solverosqp.qp_time*1e3)
        osqp_dx.append(np.array(solverosqp.dx))
        osqp_du.append(np.array(solverosqp.du))

for iter in range(1, MAX_QP_ITER_HPIPM+1):
    print(iter, " / ", MAX_QP_ITER_HPIPM)
    # HPIPM_DENSE
    if('HPIPM_DENSE' in SOLVERS):
        print("   Problem :  HPIPM_DENSE")
        solverhpipm_dense.max_qp_iters = iter
        solverhpipm_dense.xs = [solverhpipm_dense.problem.x0] * (solverhpipm_dense.problem.T + 1) 
        solverhpipm_dense.us = solverhpipm_dense.problem.quasiStatic([solverhpipm_dense.problem.x0] * solverhpipm_dense.problem.T)
        solverhpipm_dense.solve(solverhpipm_dense.xs, solverhpipm_dense.us, 1, False)
        hpipm_time.append(solverhpipm_dense.qp_time*1e3)
        hpipm_dx.append(np.array(solverhpipm_dense.dx))
        hpipm_du.append(np.array(solverhpipm_dense.du))

    # HPIPM_OCP    
    if('HPIPM_OCP' in SOLVERS):    
        print("   Problem :  HPIPM_OCP")
        solverhpipm_ocp.max_qp_iters = iter
        solverhpipm_ocp.xs = [solverhpipm_ocp.problem.x0] * (solverhpipm_ocp.problem.T + 1) 
        solverhpipm_ocp.us = solverhpipm_ocp.problem.quasiStatic([solverhpipm_ocp.problem.x0] * solverhpipm_ocp.problem.T)
        solverhpipm_ocp.solve(solverhpipm_ocp.xs, solverhpipm_ocp.us, 1, False)
        hpipm_ocp_time.append(solverhpipm_ocp.qp_time*1e3)
        # hpipm_ocp_time.append(iter)
        hpipm_ocp_dx.append(np.array(solverhpipm_ocp.dx))
        hpipm_ocp_du.append(np.array(solverhpipm_ocp.du))



csqp_time = np.array(csqp_time)
osqp_time = np.array(osqp_time)
hpipm_time = np.array(hpipm_time)
hpipm_ocp_time = np.array(hpipm_ocp_time)

# def distance(x, y):
#     # print(x.shape)
#     return np.linalg.norm(x[0] - y[0], 2)


def distance(x, y):
    # print(x.shape)
    return np.linalg.norm(x - y, 2)

csqp_distance_to_GT_x = np.array([distance(dx, dx_GT) for dx in csqp_dx])
osqp_distance_to_GT_x = np.array([distance(dx, dx_GT) for dx in osqp_dx])
hpipm_distance_to_GT_x = np.array([distance(dx, dx_GT) for dx in hpipm_dx])
hpipm_ocp_distance_to_GT_x = np.array([distance(dx, dx_GT) for dx in hpipm_ocp_dx])

csqp_distance_to_GT_u = np.array([distance(du, du_GT) for du in csqp_du])
osqp_distance_to_GT_u = np.array([distance(du, du_GT) for du in osqp_du])
hpipm_distance_to_GT_u = np.array([distance(du, du_GT) for du in hpipm_du])
hpipm_ocp_distance_to_GT_u = np.array([distance(du, du_GT) for du in hpipm_ocp_du])

csqp_distance_to_GT =  csqp_distance_to_GT_x + csqp_distance_to_GT_u
osqp_distance_to_GT = osqp_distance_to_GT_x + osqp_distance_to_GT_u
hpipm_distance_to_GT = hpipm_distance_to_GT_x + hpipm_distance_to_GT_u
hpipm_ocp_distance_to_GT = hpipm_ocp_distance_to_GT_x + hpipm_ocp_distance_to_GT_u

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
 
fig0, ax0 = plt.subplots(1, 1, figsize=(19.2,10.8))
if('CSQP' in SOLVERS): 
    ax0.plot(csqp_time, csqp_distance_to_GT, color='r', linestyle='solid', linewidth=4, label='CSQP') 
if('OSQP' in SOLVERS): 
    ax0.plot(osqp_time, osqp_distance_to_GT, color='y', linestyle='dashed', linewidth=4, label='OSQP') 
if('HPIPM_DENSE' in SOLVERS): 
    ax0.plot(hpipm_time, hpipm_distance_to_GT, color='g', linestyle='dotted',linewidth=4, label='HPIPM (dense)') 
if('HPIPM_OCP' in SOLVERS): 
    ax0.plot(hpipm_ocp_time, hpipm_ocp_distance_to_GT, color='b', linestyle='dashdot', linewidth=4, label='HPIPM (OCP)') #marker='o', markerfacecolor='b', linestyle='-', markersize=12, markeredgecolor='k', alpha=1., label='SQP')

# Set axis and stuff
ax0.set_xlabel('Time [ms]', fontsize=26)
ax0.set_ylabel(r'$\vert\vert x - x^{\star}\vert\vert$', fontsize=26)
ax0.set_yscale("log")
ax0.tick_params(axis = 'y', labelsize=22)
ax0.tick_params(axis = 'x', labelsize=22)
ax0.grid(True) 
# Legend 
handles0, labels0 = ax0.get_legend_handles_labels()
fig0.legend(handles0, labels0, loc='lower right', bbox_to_anchor=(0.902, 0.1), prop={'size': 26}) 
plt.show()