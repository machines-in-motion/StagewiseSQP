'''
Compare constrained QP solvers timings :
- CSQP a.k.a. stagewise ADMM : sequential     + dense algebra
- OSQP                       : not sequential + sparse algebra
- HPIPM_dense                : not sequential + dense algebra
- HPIPM_ocp                  : sequential     + dense algebra
On the following constrained OCPs:
- kuka
- quadrotor
- double pendulum

>>> becnhmarks for rebutal :
    - osqp dense vs stagewise (ours)
    - reply "look at S. Caron osqp (dense) > hpipm (dense) to reviewer
    - remove any mention to hpipm in the paper
Randomizing over initial states
'''
import numpy as np
import pinocchio as pin
from problems import create_solo12_problem, create_kuka_problem, create_humanoid_taichi_problem, create_quadrotor_problem

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

# Solver params
MAXITER     = 1     
TOL         = 1e-4
CALLBACKS   = False
MAX_QP_ITER = 5000
EPS_ABS     = 1e-4
EPS_REL     = 0.
SAVE        = True # Save figure 


# Benchmark params
SEED = 10 ; np.random.seed(SEED)
N_samples = 100


# name = 'solo12'
# name = 'Kuka'
name = 'Taichi'
        

# Solvers
SOLVERS = ['CSQP',
           'OSQP',
           'HPIPM_DENSE', 
           'HPIPM_OCP']

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

# Initial states


# Create 1 solver of each type for each problem
print('------')

if(name == "solo12"):  
    mu = 0.8
    pb = create_solo12_problem(mu)
    solo12                 = load_pinocchio_wrapper("solo12")
    ee_frame_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
    solo12_mu_samples  = np.zeros((N_samples))
if(name == "Kuka"):      
    kuka_x0      = np.zeros(14)
    pb = create_kuka_problem(kuka_x0) 
    kuka                 = load_pinocchio_wrapper("iiwa")
    kuka_data = kuka.model.createData()
    kuka_x0_samples      = np.zeros((N_samples, kuka.model.nq + kuka.model.nv))
if(name == "Quadrotor"): 
    quadrotor    = example_robot_data.load('hector') 
    quadrotor_x0 = np.array(list(quadrotor.q0) + [0.]*quadrotor.model.nv) 
    pb = create_quadrotor_problem(quadrotor_x0) 
    quadrotor            = example_robot_data.load('hector') 
    quadrotor_x0_samples = np.zeros((N_samples, quadrotor.model.nq + quadrotor.model.nv))
if(name == "Taichi"): 
    taichi_p0    = np.array([0.4, 0, 1.2])
    pb = create_humanoid_taichi_problem(taichi_p0) 
    taichi_p0_samples  = np.zeros((N_samples, 3))


# Initial state samples
for i in range(N_samples):
    if(name == "solo12"):  
        solo12_mu_samples[i]  = 0.8 + 0.1 * (2*np.random.rand() - 1)
    if(name == "Kuka"):  
        kuka_x0_samples[i,:]      = np.concatenate([pin.randomConfiguration(kuka.model), np.random.random(kuka.model.nv)])
    if(name == "Quadrotor"): 
        quadrotor_x0_samples[i,:] = np.concatenate([pin.randomConfiguration(quadrotor.model), np.zeros(quadrotor.model.nv)])
    if(name == "Taichi"): 
        err = np.array([0., 0., 2*np.random.rand() - 1])
        taichi_p0_samples[i,:]  = taichi_p0 + 0.05*err



# Create solver CSQP 
if('CSQP' in SOLVERS):
    solvercsqp = mim_solvers.SolverCSQP(pb)
    solvercsqp.termination_tolerance = TOL
    solvercsqp.max_qp_iters = MAX_QP_ITER
    solvercsqp.with_qp_callbacks = False
    solvercsqp.eps_abs = EPS_ABS
    solvercsqp.eps_rel = EPS_REL
    solvercsqp.equality_qp_initial_guess = False
    solvercsqp.update_rho_with_heuristic = False
    solvercsqp.with_callbacks = CALLBACKS
    # solvercsqp.with_qp_callbacks = True # CALLBACKS

# Create solver OSQP
if('OSQP' in SOLVERS):
    solverosqp = CSQP(pb, "OSQP")
    solverosqp.termination_tolerance = TOL
    solverosqp.max_qp_iters = MAX_QP_ITER
    solverosqp.eps_abs = EPS_ABS
    solverosqp.eps_rel = EPS_REL
    solverosqp.with_callbacks = CALLBACKS

# Create solver HPIPM dense
if('HPIPM_DENSE' in SOLVERS):
    solverhpipm_dense = CSQP(pb, "HPIPM_DENSE")
    solverhpipm_dense.termination_tolerance  = TOL
    solverhpipm_dense.max_qp_iters = MAX_QP_ITER
    solverhpipm_dense.eps_abs = EPS_ABS
    solverhpipm_dense.eps_rel = EPS_REL
    solverhpipm_dense.with_callbacks = CALLBACKS

# Create solver HPIPM ocp
if('HPIPM_OCP' in SOLVERS):
    solverhpipm_ocp = CSQP(pb, "HPIPM_OCP")
    solverhpipm_ocp.termination_tolerance  = TOL
    solverhpipm_ocp.max_qp_iters = MAX_QP_ITER
    solverhpipm_ocp.eps_abs = EPS_ABS
    solverhpipm_ocp.eps_rel = EPS_REL
    solverhpipm_ocp.with_callbacks = CALLBACKS


print("Created "+str(N_samples)+" random initial states per model !")

# Solve problems for sample initial states
for sample_id in range(N_samples):
    
    print("---")
    print("Sample "+str(sample_id+1)+'/'+str(N_samples))

    # Initial state
    if(name == "Cartpole"):  x0 = cartpole_x0_samples[i,:]
    if(name == "Kuka"):      x0 = kuka_x0_samples[i,:]
    if(name == "Quadrotor"): x0 = quadrotor_x0_samples[i,:]
    if(name == "Taichi"):    p0 = taichi_p0_samples[i,:]
    if(name == "solo12"):    mu = solo12_mu_samples[i]

    # CSQP
    if('CSQP' in SOLVERS):
        print("   Problem : "+name+" CSQP")
        if(name == "solo12"):
            models = list(solvercsqp.problem.runningModels)
            for m in models: 
                for name_ee in ee_frame_names:
                    name_constraint = solo12.model.frames[solo12.model.getFrameId(name_ee)].name + "_contact"
                    m.differential.constraints.constraints[name_constraint+"friction"].constraint.residual.mu = mu
        elif(name == "Taichi"):
            models = list(solvercsqp.problem.runningModels) + [solvercsqp.problem.terminalModel]
            for m in models: m.differential.costs.costs["gripperPose"].cost.residual.reference = pin.SE3(np.eye(3), p0.copy())
        else:
            solvercsqp.problem.x0 = x0
        solvercsqp.xs = [solvercsqp.problem.x0] * (solvercsqp.problem.T + 1) 
        solvercsqp.us = solvercsqp.problem.quasiStatic([solvercsqp.problem.x0] * solvercsqp.problem.T)
        solvercsqp.solve(solvercsqp.xs, solvercsqp.us, 0, False)
        t1 = time.time()
        solvercsqp.computeDirection(True)
        solvercsqp.qp_time = time.time() - t1
        # Check convergence and fill data 
        solved = (solvercsqp.norm_primal < EPS_ABS and solvercsqp.norm_dual < EPS_ABS and solvercsqp.qp_iters <= MAX_QP_ITER)
        csqp_solved_samples.append( solved )
        if(not solved): 
            print("      FAILED !!!!")
            csqp_iter_samples.append(np.inf)
            csqp_time_samples.append(np.inf)
        else:
            csqp_iter_samples.append(solvercsqp.qp_iters)
            csqp_time_samples.append(solvercsqp.qp_time*1e3)
            print("     QP Time = ", solvercsqp.qp_time)
            print("     QP Iter = ", solvercsqp.qp_iters)
        # print(" - Primal residual: ", solvercsqp.norm_primal)
        # print(" - Dual residual: ", solvercsqp.norm_dual)
    
    # OSQP
    if('OSQP' in SOLVERS):
        print("   Problem : "+name+" OSQP")
        if(name == "solo12"):
            models = list(solverosqp.problem.runningModels)
            for m in models: 
                for name_ee in ee_frame_names:
                    name_constraint = solo12.model.frames[solo12.model.getFrameId(name_ee)].name + "_contact"
                    m.differential.constraints.constraints[name_constraint+"friction"].constraint.residual.mu = mu
        elif(name == "Taichi"):
            models = list(solvercsqp.problem.runningModels) + [solvercsqp.problem.terminalModel]
            for m in models: m.differential.costs.costs["gripperPose"].cost.residual.reference = pin.SE3(np.eye(3), p0.copy())
        else:
            solvercsqp.problem.x0 = x0
        solverosqp.xs = [solverosqp.problem.x0] * (solverosqp.problem.T + 1) 
        solverosqp.us = solverosqp.problem.quasiStatic([solverosqp.problem.x0] * solverosqp.problem.T)
        solverosqp.solve(solverosqp.xs, solverosqp.us, MAXITER, False)
        # solved = (solverosqp.found_qp_sol and solverosqp.norm_primal < EPS_ABS and solverosqp.norm_dual < EPS_ABS and solverosqp.qp_iters <= MAX_QP_ITER)
        solved = (solverosqp.norm_primal < EPS_ABS and solverosqp.norm_dual < EPS_ABS and solverosqp.qp_iters <= MAX_QP_ITER)
        osqp_solved_samples.append( solved )
        if(not solved): 
            print("      FAILED !!!!")
            osqp_iter_samples.append(np.inf)
            osqp_time_samples.append(np.inf)
        else:
            osqp_iter_samples.append(solverosqp.qp_iters)
            osqp_time_samples.append(solverosqp.qp_time*1e3)
            print("     QP Time = ", solverosqp.qp_time)
            print("     QP Iter = ", solverosqp.qp_iters)

    # HPIPM_DENSE
    if('HPIPM_DENSE' in SOLVERS):
        print("   Problem : "+name+" HPIPM_DENSE")
        if(name == "solo12"):
            models = list(solverhpipm_dense.problem.runningModels)
            for m in models: 
                for name_ee in ee_frame_names:
                    name_constraint = solo12.model.frames[solo12.model.getFrameId(name_ee)].name + "_contact"
                    m.differential.constraints.constraints[name_constraint+"friction"].constraint.residual.mu = mu
        elif(name == "Taichi"):
            models = list(solvercsqp.problem.runningModels) + [solvercsqp.problem.terminalModel]
            for m in models: m.differential.costs.costs["gripperPose"].cost.residual.reference = pin.SE3(np.eye(3), p0.copy())
        else:
            solvercsqp.problem.x0 = x0
        solverhpipm_dense.xs = [solverhpipm_dense.problem.x0] * (solverhpipm_dense.problem.T + 1) 
        solverhpipm_dense.us = solverhpipm_dense.problem.quasiStatic([solverhpipm_dense.problem.x0] * solverhpipm_dense.problem.T)
        solverhpipm_dense.solve(solverhpipm_dense.xs, solverhpipm_dense.us, MAXITER, False)
        solved = (solverhpipm_dense.norm_primal < EPS_ABS and solverhpipm_dense.norm_dual < EPS_ABS and solverhpipm_dense.qp_iters <= MAX_QP_ITER)
        # solverhpipm_dense.found_qp_sol = False
        # if(solverhpipm_dense.found_qp_sol):
        #     solved = (solverhpipm_dense.norm_primal < EPS_ABS and solverhpipm_dense.norm_dual < EPS_ABS and solverhpipm_dense.qp_iters <= MAX_QP_ITER)
        # else:
        #     solved = False
        # solved = False
        hpipm_dense_solved_samples.append( solved )
        if(not solved): 
            print("      FAILED !!!!")
            hpipm_dense_iter_samples.append(np.inf)
            hpipm_dense_time_samples.append(np.inf)
        else:
            hpipm_dense_iter_samples.append(solverhpipm_dense.qp_iters)
            hpipm_dense_time_samples.append(solverhpipm_dense.qp_time*1e3)
            print("     QP Time = ", solverhpipm_dense.qp_time)
            print("     QP Iter = ", solverhpipm_dense.qp_iters)

    # HPIPM_OCP    
    if('HPIPM_OCP' in SOLVERS):    
        print("   Problem : "+name+" HPIPM_OCP")
        if(name == "solo12"):
            models = list(solverhpipm_ocp.problem.runningModels)
            for m in models: 
                for name_ee in ee_frame_names:
                    name_constraint = solo12.model.frames[solo12.model.getFrameId(name_ee)].name + "_contact"
                    m.differential.constraints.constraints[name_constraint+"friction"].constraint.residual.mu = mu
        elif(name == "Taichi"):
            models = list(solvercsqp.problem.runningModels) + [solvercsqp.problem.terminalModel]
            for m in models: m.differential.costs.costs["gripperPose"].cost.residual.reference = pin.SE3(np.eye(3), p0.copy())
        else:
            solvercsqp.problem.x0 = x0
        solverhpipm_ocp.xs = [solverhpipm_ocp.problem.x0] * (solverhpipm_ocp.problem.T + 1) 
        solverhpipm_ocp.us = solverhpipm_ocp.problem.quasiStatic([solverhpipm_ocp.problem.x0] * solverhpipm_ocp.problem.T)
        solverhpipm_ocp.solve(solverhpipm_ocp.xs, solverhpipm_ocp.us, MAXITER, False)
            # Check convergence
        solved = (solverhpipm_ocp.norm_primal < EPS_ABS and solverhpipm_ocp.norm_dual < EPS_ABS and solverhpipm_ocp.qp_iters <= MAX_QP_ITER)
        # if(solverhpipm_ocp.found_qp_sol):
        #     solved = (solverhpipm_ocp.norm_primal < EPS_ABS and solverhpipm_ocp.norm_dual < EPS_ABS and solverhpipm_ocp.qp_iters <= MAX_QP_ITER)
        # else:
        #     solved = False
        hpipm_ocp_solved_samples.append( solved )
        if(not solved): 
            print("      FAILED !!!!")
            hpipm_ocp_iter_samples.append(np.inf)
            hpipm_ocp_time_samples.append(np.inf)
        else:
            hpipm_ocp_iter_samples.append(solverhpipm_ocp.qp_iters)
            hpipm_ocp_time_samples.append(solverhpipm_ocp.qp_time*1e3)
        print("     QP Time = ", solverhpipm_ocp.qp_time)
        print("     QP Iter = ", solverhpipm_ocp.qp_iters)

if(SAVE):
    PREFIX = 'data/'
    file_name = PREFIX + name + "_qp_benchmark"
    np.savez(file_name, 
            csqp_iter_solved_samples=csqp_solved_samples, 
            csqp_time_samples=csqp_time_samples,
            csqp_iter_samples=csqp_iter_samples,
            osqp_iter_solved_samples=osqp_solved_samples, 
            osqp_time_samples=osqp_time_samples,
            osqp_iter_samples=osqp_iter_samples,
            hpipm_dense_solved_samples=hpipm_dense_solved_samples, 
            hpipm_dense_time_samples=hpipm_dense_time_samples,
            hpipm_dense_iter_samples=hpipm_dense_iter_samples,
            hpipm_ocp_solved_samples=hpipm_ocp_solved_samples, 
            hpipm_ocp_time_samples=hpipm_ocp_time_samples,
            hpipm_ocp_iter_samples=hpipm_ocp_iter_samples)
    PREFIX = '/home/skleff/SQP_REBUTAL_BENCH/'
    file_name = name + "_qp_benchmark"
    np.savez(file_name, 
            csqp_iter_solved_samples=csqp_solved_samples, 
            csqp_time_samples=csqp_time_samples,
            csqp_iter_samples=csqp_iter_samples,
            osqp_iter_solved_samples=osqp_solved_samples, 
            osqp_time_samples=osqp_time_samples,
            osqp_iter_samples=osqp_iter_samples,
            hpipm_dense_solved_samples=hpipm_dense_solved_samples, 
            hpipm_dense_time_samples=hpipm_dense_time_samples,
            hpipm_dense_iter_samples=hpipm_dense_iter_samples,
            hpipm_ocp_solved_samples=hpipm_ocp_solved_samples, 
            hpipm_ocp_time_samples=hpipm_ocp_time_samples,
            hpipm_ocp_iter_samples=hpipm_ocp_iter_samples)
