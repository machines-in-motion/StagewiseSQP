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
SOLVERS = ['CSQP']
        #    'OSQP']
        #    'HPIPM_DENSE', 
        #    'HPIPM_OCP']

TOL         = 0.
CALLBACKS   = False
EPS_IP      = 1e-10
EPS_REL     = 0.
SAVE = True

name = "Kuka"
# name = "solo12"
# name = "Taichi"



if name == "solo12":
    MAX_QP_ITER_HPIPM = 10 
    MAX_QP_ITER_ADMM  = 100


if name == "Taichi":
    MAX_QP_ITER_HPIPM = 50
    MAX_QP_ITER_ADMM  = 500 #250

if name == "Kuka":
    MAX_QP_ITER_HPIPM = 10 
    MAX_QP_ITER_ADMM  = 200 
    kuka = load_pinocchio_wrapper("iiwa")

SEED = 10 ; np.random.seed(SEED)
n_samples = 100

def distance(x, y):
    return np.linalg.norm(x - y, np.inf)

csqp_time = []
csqp_iter = []
csqp_dist = []

osqp_time = []
osqp_iter = []
osqp_dist = []

hpipm_ocp_time = []
hpipm_ocp_dist = []


# n_samples = 1000

for sample_id in range(n_samples):
    print("\n Sample ID = ", sample_id)

    if(name == "Kuka"):
        q0 = np.array([0., 1.04, 0., -1.13, 0.2,  0.78, 0])     # Initial robot joint configuration
        kuka_x0 = np.concatenate([q0 + 2 * (np.random.random(kuka.model.nv) - 0.5), 2 * (np.random.random(kuka.model.nv) - 0.5)])
        pb = create_kuka_problem(kuka_x0) 

    if(name == "solo12"):
        mu = 0.8 + 0.01 * (2*np.random.rand() - 1)
        pb = create_solo12_problem(mu)

    if(name == "Taichi"):
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

    # Create solver CSQP 
    if('OSQP' in SOLVERS):
        solverosqp = CSQP(pb, "OSQP")
        solverosqp.termination_tolerance = TOL
        solverosqp.with_qp_callbacks = False
        solverosqp.OSQP_scaling = True
        solverosqp.eps_abs = 1e-100
        solverosqp.eps_rel = EPS_REL
        solverosqp.with_callbacks = CALLBACKS

    # Create solver HPIPM ocp
    if('HPIPM_OCP' in SOLVERS):
        solverhpipm_ocp = CSQP(pb, "HPIPM_OCP")
        solverhpipm_ocp.termination_tolerance  = TOL
        solverhpipm_ocp.eps_abs = EPS_IP
        solverhpipm_ocp.eps_rel = EPS_REL
        solverhpipm_ocp.with_callbacks = CALLBACKS


    # Ground truth 
    solverGT = mim_solvers.SolverCSQP(pb) #CSQP(pb, "HPIPM_OCP")
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
    print("qp_iter GT = ", solverGT.qp_iters)

    dx_GT = np.array(solverGT.dx, copy=True)
    du_GT = np.array(solverGT.du, copy=True)

    dx = np.zeros_like(dx_GT)
    du = np.zeros_like(du_GT)
    dist_HPIPM = distance(dx, dx_GT) + distance(du, du_GT)
    dist_CSQP = distance(dx, dx_GT) + distance(du, du_GT)
    dist_OSQP = distance(dx, dx_GT) + distance(du, du_GT)

    sample_csqp_time = [0.]
    sample_csqp_iter = [0.]
    sample_csqp_dist = [dist_CSQP]

    sample_osqp_time = [0.]
    sample_osqp_iter = [0.]
    sample_osqp_dist = [dist_OSQP]

    sample_hpipm_ocp_time = [0.]
    sample_hpipm_ocp_dist = [dist_HPIPM]
    for iter in range(1, MAX_QP_ITER_ADMM+1, 1):
        # print(iter, " / ", MAX_QP_ITER_ADMM)
        # CSQP
        if('CSQP' in SOLVERS):
            solvercsqp.max_qp_iters = iter
            solvercsqp.xs = [solvercsqp.problem.x0] * (solvercsqp.problem.T + 1) 
            solvercsqp.us = solvercsqp.problem.quasiStatic([solvercsqp.problem.x0] * solvercsqp.problem.T)
            solvercsqp.solve(solvercsqp.xs, solvercsqp.us, 0, False)
            t1 = time.time()
            solvercsqp.computeDirection(True)
            solvercsqp.qp_time = time.time() - t1
            dx = np.array(solvercsqp.dx)
            du = np.array(solvercsqp.du)
            dist = distance(dx, dx_GT) + distance(du, du_GT)
            sample_csqp_time.append(solvercsqp.qp_time*1e3)
            sample_csqp_dist.append(dist)
            sample_csqp_iter.append(solvercsqp.qp_iters)

        # OSQP
        if('OSQP' in SOLVERS):
            solverosqp.max_qp_iters = iter
            solverosqp.xs = [solverosqp.problem.x0] * (solverosqp.problem.T + 1) 
            solverosqp.us = solverosqp.problem.quasiStatic([solverosqp.problem.x0] * solverosqp.problem.T)
            solverosqp.solve(solverosqp.xs, solverosqp.us, 1, False)
            dx = np.array(solverosqp.dx)
            du = np.array(solverosqp.du)
            dist = distance(dx, dx_GT) + distance(du, du_GT)
            sample_osqp_time.append(solverosqp.qp_time*1e3)
            sample_osqp_dist.append(dist)
            sample_osqp_iter.append(solverosqp.qp_iters)

    for iter in range(1, MAX_QP_ITER_HPIPM+1):
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
    csqp_iter.append(sample_csqp_iter)
    csqp_dist.append(sample_csqp_dist)

    osqp_time.append(sample_osqp_time)
    osqp_iter.append(sample_osqp_iter)
    osqp_dist.append(sample_osqp_dist)

    hpipm_ocp_time.append(sample_hpipm_ocp_time)
    hpipm_ocp_dist.append(sample_hpipm_ocp_dist)

# convert to array
csqp_time = np.array(csqp_time)
csqp_iter = np.array(csqp_iter)
csqp_dist = np.array(csqp_dist)

osqp_time = np.array(osqp_time)
osqp_iter = np.array(osqp_iter)
osqp_dist = np.array(osqp_dist)

hpipm_ocp_time = np.array(hpipm_ocp_time)
hpipm_ocp_dist = np.array(hpipm_ocp_dist)

# Extract mean and std
mean_csqp_time = np.mean(csqp_time, axis=0)
mean_csqp_iter = np.mean(csqp_iter, axis=0)
mean_osqp_time = np.mean(osqp_time, axis=0)
mean_osqp_iter = np.mean(osqp_iter, axis=0)
mean_hpipm_ocp_time = np.mean(hpipm_ocp_time, axis=0)

mean_csqp_dist = np.median(csqp_dist, axis=0)
mean_osqp_dist = np.median(osqp_dist, axis=0)
mean_hpipm_ocp_dist = np.median(hpipm_ocp_dist, axis=0)

q25_csqp_dist = np.quantile(csqp_dist, 0.25, axis=0)
q25_osqp_dist = np.quantile(osqp_dist, 0.25, axis=0)
q25_hpipm_ocp_dist = np.quantile(hpipm_ocp_dist, 0.25, axis=0)

q75_csqp_dist = np.quantile(csqp_dist, 0.75, axis=0)
q75_osqp_dist = np.quantile(osqp_dist, 0.75, axis=0)
q75_hpipm_ocp_dist = np.quantile(hpipm_ocp_dist, 0.75, axis=0)

std_csqp_dist = np.std(csqp_dist, axis=0)
std_osqp_dist = np.std(osqp_dist, axis=0)
std_hpipm_ocp_dist = np.std(hpipm_ocp_dist, axis=0)

if(SAVE):
    PREFIX = 'data/'
    file_name = PREFIX + name + "_qp_convergence"
    print("saving to "+file_name)
    np.savez(file_name, 
            csqp_time=csqp_time, 
            csqp_iter=csqp_iter, 
            csqp_dist=csqp_dist,
            osqp_time=osqp_time, 
            osqp_iter=osqp_iter, 
            osqp_dist=osqp_dist)
    PREFIX = '/home/skleff/SQP_REBUTAL_BENCH/constrained/'
    file_name = PREFIX + name + "_qp_convergence"
    np.savez(file_name, 
            csqp_time=csqp_time, 
            csqp_iter=csqp_iter, 
            csqp_dist=csqp_dist,            
            osqp_time=osqp_time, 
            osqp_iter=osqp_iter, 
            osqp_dist=osqp_dist)