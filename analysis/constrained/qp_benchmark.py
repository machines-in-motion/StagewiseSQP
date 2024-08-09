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
MAX_QP_ITER = 100000
EPS_ABS     = 1e-2
EPS_REL     = 0.
SAVE        = True # Save figure 


# Benchmark params
SEED = 10 ; np.random.seed(SEED)
N_samples = 10
names = [
        'solo12']      # maxiter = 500 TIME_DISCRETIZATION=0.1 TOL=1e-4 MAX_QP_TIME=1e5
        #  'Kuka']       # maxiter = 100
        #  'Taichi']       #
        #  'Quadrotor']  # maxiter = 200
if('Taichi' in names):
    MAX_QP_TIME = int(1e5)     # in ms
    TIME_DISCRETIZATION = 1.  # the larger the faster (usefull for very fast problems) 
if('Kuka' in names):
    MAX_QP_TIME = int(1e2)       # in ms
    TIME_DISCRETIZATION = 0.01  # the larger the faster (usefull for very fast problems) 
if('solo12' in names):
    MAX_QP_TIME = int(2e3)     # in ms
    TIME_DISCRETIZATION = 0.01  # the larger the faster (usefull for very fast problems) 

N_pb = len(names)

# Solvers
SOLVERS = ['CSQP',
           'OSQP',
        #    'HPIPM_DENSE', 
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
pendulum_x0  = np.array([3.14, 0., 0., 0.])
cartpole_x0  = np.array([0., 3.14, 0., 0.])
kuka_x0      = np.zeros(14)
quadrotor    = example_robot_data.load('hector') 
quadrotor_x0 = np.array(list(quadrotor.q0) + [0.]*quadrotor.model.nv) 
taichi_p0    = np.array([0.4, 0, 1.2])

# Create 1 solver of each type for each problem
print('------')
for k,name in enumerate(names):
    if(name == "solo12"):  
        pb = create_solo12_problem(pendulum_x0)
    if(name == "Cartpole"):  
        pb = create_cartpole_problem(cartpole_x0) 
    if(name == "Kuka"):      
        pb = create_kuka_problem(kuka_x0) 
    if(name == "Quadrotor"): 
        pb = create_quadrotor_problem(quadrotor_x0) 
    if(name == "Taichi"): 
        pb = create_humanoid_taichi_problem(taichi_p0) 

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
        solversCSQP.append(solvercsqp)
    
    # Create solver OSQP
    if('OSQP' in SOLVERS):
        solverosqp = CSQP(pb, "OSQP")
        solverosqp.termination_tolerance = TOL
        solverosqp.max_qp_iters = MAX_QP_ITER
        solverosqp.eps_abs = EPS_ABS
        solverosqp.eps_rel = EPS_REL
        solverosqp.with_callbacks = CALLBACKS
        solversOSQP.append(solverosqp)

    # Create solver HPIPM dense
    if('HPIPM_DENSE' in SOLVERS):
        solverhpipm_dense = CSQP(pb, "HPIPM_DENSE")
        solverhpipm_dense.termination_tolerance  = TOL
        solverhpipm_dense.max_qp_iters = MAX_QP_ITER
        solverhpipm_dense.eps_abs = EPS_ABS
        solverhpipm_dense.eps_rel = EPS_REL
        solverhpipm_dense.with_callbacks = CALLBACKS
        solversHPIPM_dense.append(solverhpipm_dense)

    # Create solver HPIPM ocp
    if('HPIPM_OCP' in SOLVERS):
        solverhpipm_ocp = CSQP(pb, "HPIPM_OCP")
        solverhpipm_ocp.termination_tolerance  = TOL
        solverhpipm_ocp.max_qp_iters = MAX_QP_ITER
        solverhpipm_ocp.eps_abs = EPS_ABS
        solverhpipm_ocp.eps_rel = EPS_REL
        solverhpipm_ocp.with_callbacks = CALLBACKS
        solversHPIPM_ocp.append(solverhpipm_ocp)



# Initial state samples
solo12                 = load_pinocchio_wrapper("solo12")
ee_frame_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
cartpole_x0_samples  = np.zeros((N_samples, 4))
kuka                 = load_pinocchio_wrapper("iiwa")
kuka_data = kuka.model.createData()
kuka_x0_samples      = np.zeros((N_samples, kuka.model.nq + kuka.model.nv))
quadrotor            = example_robot_data.load('hector') 
quadrotor_x0_samples = np.zeros((N_samples, quadrotor.model.nq + quadrotor.model.nv))
taichi_p0_samples  = np.zeros((N_samples, 3))
solo12_mu_samples  = np.zeros((N_samples))


for i in range(N_samples):
    cartpole_x0_samples[i,:]  = np.array([0., np.pi/2, 0., 0.])
    kuka_x0_samples[i,:]      = np.concatenate([pin.randomConfiguration(kuka.model), np.random.random(kuka.model.nv)])
    quadrotor_x0_samples[i,:] = np.concatenate([pin.randomConfiguration(quadrotor.model), np.zeros(quadrotor.model.nv)])
    err = np.zeros(3)
    err[2] = 2*np.random.rand() - 1
    taichi_p0_samples[i,:]  = taichi_p0 + 0.05*err
    solo12_mu_samples[i]  = 0.8 + 0.01 * (2*np.random.rand() - 1)

print("Created "+str(N_samples)+" random initial states per model !")

# Solve problems for sample initial states
for i in range(N_samples):
    if('CSQP' in SOLVERS):
        csqp_iter_samples.append([])
        csqp_time_samples.append([])
        csqp_solved_samples.append([])

    if('OSQP' in SOLVERS):
        osqp_iter_samples.append([])
        osqp_time_samples.append([])
        osqp_solved_samples.append([])

    if('HPIPM_DENSE' in SOLVERS):
        hpipm_dense_iter_samples.append([])
        hpipm_dense_time_samples.append([])
        hpipm_dense_solved_samples.append([])

    if('HPIPM_OCP' in SOLVERS):
        hpipm_ocp_iter_samples.append([])
        hpipm_ocp_time_samples.append([])
        hpipm_ocp_solved_samples.append([])

    print("---")
    print("Sample "+str(i+1)+'/'+str(N_samples))
    for k,name in enumerate(names):
        # Initial state
        if(name == "Cartpole"):  x0 = cartpole_x0_samples[i,:]
        if(name == "Kuka"):      x0 = kuka_x0_samples[i,:]
        if(name == "Quadrotor"): x0 = quadrotor_x0_samples[i,:]
        if(name == "Taichi"):    p0 = taichi_p0_samples[i,:]
        if(name == "solo12"):    mu = solo12_mu_samples[i]

        # CSQP
        if('CSQP' in SOLVERS):
            print("   Problem : "+name+" CSQP")
            solvercsqp = solversCSQP[k]
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
            csqp_solved_samples[i].append( solved )
            if(not solved): 
                print("      FAILED !!!!")
                csqp_iter_samples[i].append(MAX_QP_ITER)
                csqp_time_samples[i].append(MAX_QP_TIME+1)
            else:
                csqp_iter_samples[i].append(solvercsqp.qp_iters)
                csqp_time_samples[i].append(solvercsqp.qp_time*1e3)
                print("     QP Time = ", solvercsqp.qp_time)
                print("     QP Iter = ", solvercsqp.qp_iters)
            # print(" - Primal residual: ", solvercsqp.norm_primal)
            # print(" - Dual residual: ", solvercsqp.norm_dual)
        
        # OSQP
        if('OSQP' in SOLVERS):
            print("   Problem : "+name+" OSQP")
            solverosqp = solversOSQP[k]
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
            osqp_solved_samples[i].append( solved )
            if(not solved): 
                print("      FAILED !!!!")
                osqp_iter_samples[i].append(MAX_QP_ITER)
                osqp_time_samples[i].append(MAX_QP_TIME+1)
            else:
                osqp_iter_samples[i].append(solverosqp.qp_iters)
                osqp_time_samples[i].append(solverosqp.qp_time*1e3)
                print("     QP Time = ", solverosqp.qp_time)
                print("     QP Iter = ", solverosqp.qp_iters)

        # HPIPM_DENSE
        if('HPIPM_DENSE' in SOLVERS):
            print("   Problem : "+name+" HPIPM_DENSE")
            solverhpipm_dense = solversHPIPM_dense[k]
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
            hpipm_dense_solved_samples[i].append( solved )
            if(not solved): 
                print("      FAILED !!!!")
                hpipm_dense_iter_samples[i].append(MAX_QP_ITER)
                hpipm_dense_time_samples[i].append(MAX_QP_TIME+1)
            else:
                hpipm_dense_iter_samples[i].append(solverhpipm_dense.qp_iters)
                hpipm_dense_time_samples[i].append(solverhpipm_dense.qp_time*1e3)
                print("     QP Time = ", solverhpipm_dense.qp_time)
                print("     QP Iter = ", solverhpipm_dense.qp_iters)

        # HPIPM_OCP    
        if('HPIPM_OCP' in SOLVERS):    
            print("   Problem : "+name+" HPIPM_OCP")
            solverhpipm_ocp = solversHPIPM_ocp[k]
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
            hpipm_ocp_solved_samples[i].append( solved )
            if(not solved): 
                print("      FAILED !!!!")
                hpipm_ocp_iter_samples[i].append(MAX_QP_ITER)
                hpipm_ocp_time_samples[i].append(MAX_QP_TIME+1)
            else:
                hpipm_ocp_iter_samples[i].append(solverhpipm_ocp.qp_iters)
                hpipm_ocp_time_samples[i].append(solverhpipm_ocp.qp_time*1e3)
            print("     QP Time = ", solverhpipm_ocp.qp_time)
            print("     QP Iter = ", solverhpipm_ocp.qp_iters)

# Compute convergence statistics
TIME_VECTOR_SIZE = int((MAX_QP_TIME + 1) / TIME_DISCRETIZATION)

if('CSQP' in SOLVERS):  
    csqp_iter_solved = np.zeros((MAX_QP_ITER, N_pb))
    csqp_time_solved = np.zeros((TIME_VECTOR_SIZE, N_pb))
if('OSQP' in SOLVERS):  
    osqp_iter_solved = np.zeros((MAX_QP_ITER, N_pb))
    osqp_time_solved = np.zeros((TIME_VECTOR_SIZE, N_pb))
if('HPIPM_DENSE' in SOLVERS):  
    hpipm_dense_iter_solved = np.zeros((MAX_QP_ITER, N_pb))
    hpipm_dense_time_solved = np.zeros((TIME_VECTOR_SIZE, N_pb))
if('HPIPM_OCP' in SOLVERS):  
    hpipm_ocp_iter_solved = np.zeros((MAX_QP_ITER, N_pb))
    hpipm_ocp_time_solved = np.zeros((TIME_VECTOR_SIZE, N_pb))
for k,exp in enumerate(names):
    # Count number of problems solved for each sample initial state 
    for i in range(N_samples):
        # For sample i of problem k , compare nb iter to max iter
        if('CSQP' in SOLVERS): 
            csqp_iter_ik  = np.array(csqp_iter_samples)[i,k]
            csqp_time_ik  = np.array(csqp_time_samples)[i,k]
        if('OSQP' in SOLVERS): 
            osqp_iter_ik = np.array(osqp_iter_samples)[i,k]
            osqp_time_ik = np.array(osqp_time_samples)[i,k]
        if('HPIPM_DENSE' in SOLVERS): 
            hpipm_dense_iter_ik = np.array(hpipm_dense_iter_samples)[i,k]
            hpipm_dense_time_ik = np.array(hpipm_dense_time_samples)[i,k]
        if('HPIPM_OCP' in SOLVERS): 
            hpipm_ocp_iter_ik = np.array(hpipm_ocp_iter_samples)[i,k]
            hpipm_ocp_time_ik = np.array(hpipm_ocp_time_samples)[i,k]
        # Number of iterations
        for j in range(MAX_QP_ITER):
            if('CSQP' in SOLVERS): 
                if(csqp_iter_ik < j): csqp_iter_solved[j,k] += 1
            if('OSQP' in SOLVERS): 
                if(osqp_iter_ik < j): osqp_iter_solved[j,k] += 1
            if('HPIPM_DENSE' in SOLVERS): 
                if(hpipm_dense_iter_ik < j): hpipm_dense_iter_solved[j,k] += 1
            if('HPIPM_OCP' in SOLVERS): 
                if(hpipm_ocp_iter_ik < j): hpipm_ocp_iter_solved[j,k] += 1
        # Solve time
        for j in range(TIME_VECTOR_SIZE):
            if('CSQP' in SOLVERS): 
                if(csqp_time_ik < j * TIME_DISCRETIZATION): csqp_time_solved[j,k] += 1
            if('OSQP' in SOLVERS): 
                if(osqp_time_ik < j * TIME_DISCRETIZATION): osqp_time_solved[j,k] += 1
            if('HPIPM_DENSE' in SOLVERS): 
                if(hpipm_dense_time_ik < j * TIME_DISCRETIZATION): hpipm_dense_time_solved[j,k] += 1
            if('HPIPM_OCP' in SOLVERS): 
                if(hpipm_ocp_time_ik < j * TIME_DISCRETIZATION): hpipm_ocp_time_solved[j,k] += 1

# Generate plot of number of iterations for each problem
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
 
# x-axis : max number of iterations
xdata     = range(0,MAX_QP_ITER)

for k in range(N_pb):
    fig0, ax0 = plt.subplots(1, 1, figsize=(19.2,10.8))
    if('CSQP' in SOLVERS): 
        ax0.plot(xdata, csqp_iter_solved[:,k]/N_samples, color=COLORS['CSQP'], linestyle=LINESTYLES['CSQP'], linewidth=4, label=LABELS['CSQP']) 
    if('OSQP' in SOLVERS): 
        ax0.plot(xdata, osqp_iter_solved[:,k]/N_samples, color=COLORS['OSQP'], linestyle=LINESTYLES['OSQP'], linewidth=4, label=LABELS['OSQP'])
    if('HPIPM_DENSE' in SOLVERS): 
        ax0.plot(xdata, hpipm_dense_iter_solved[:,k]/N_samples, color=COLORS['HPIPM_DENSE'], linestyle=LINESTYLES['HPIPM_DENSE'], linewidth=4, label=LABELS['HPIPM_DENSE'])
    if('HPIPM_OCP' in SOLVERS): 
        ax0.plot(xdata, hpipm_ocp_iter_solved[:,k]/N_samples, color=COLORS['HPIPM_OCP'], linestyle=LINESTYLES['HPIPM_OCP'], linewidth=4, label=LABELS['HPIPM_OCP'])
    # Set axis and stuff
    ax0.set_ylabel('Percentage of problems solved', fontsize=26)
    ax0.set_xlabel('Max. number of iterations', fontsize=26)
    ax0.set_ylim(-0.02, 1.02)
    ax0.set_xscale('log')
    ax0.tick_params(axis = 'y', labelsize=22)
    ax0.tick_params(axis = 'x', labelsize=22)
    ax0.grid(True) 
    # Legend 
    handles0, labels0 = ax0.get_legend_handles_labels()
    fig0.legend(handles0, labels0, loc='lower right', bbox_to_anchor=(0.902, 0.1), prop={'size': 26}) 
    # Save, show , clean
    if(SAVE):
        fig0.savefig('/tmp/bench_'+names[k]+'_SEED='+str(SEED)+'_MAXITER='+str(MAXITER)+'_TOL='+str(TOL)+'.pdf', bbox_inches="tight")

# x-axis : max time allowed to solve the QP (in ms)

xdata     = np.linspace(0, MAX_QP_TIME, TIME_VECTOR_SIZE)
for k in range(N_pb):
    fig0, ax0 = plt.subplots(1, 1, figsize=(19.2,10.8))
    if('CSQP' in SOLVERS): 
        ax0.plot(xdata, csqp_time_solved[:,k]/N_samples, color=COLORS['CSQP'], linestyle=LINESTYLES['CSQP'], linewidth=4, label=LABELS['CSQP']) 
    if('OSQP' in SOLVERS): 
        ax0.plot(xdata, osqp_time_solved[:,k]/N_samples, color=COLORS['OSQP'], linestyle=LINESTYLES['OSQP'], linewidth=4, label=LABELS['OSQP'])
    if('HPIPM_DENSE' in SOLVERS): 
        ax0.plot(xdata, hpipm_dense_time_solved[:,k]/N_samples, color=COLORS['HPIPM_DENSE'], linestyle=LINESTYLES['HPIPM_DENSE'], linewidth=4, label=LABELS['HPIPM_DENSE'])
    if('HPIPM_OCP' in SOLVERS): 
        ax0.plot(xdata, hpipm_ocp_time_solved[:,k]/N_samples, color=COLORS['HPIPM_OCP'], linestyle=LINESTYLES['HPIPM_OCP'], linewidth=4, label=LABELS['HPIPM_OCP'])
    # Set axis and stuff
    ax0.set_ylabel('Percentage of problems solved', fontsize=26)
    ax0.set_xlabel('Max. solving time (ms)', fontsize=26)
    ax0.set_ylim(-0.02, 1.02)
    # ax0.set_xscale("log")
    ax0.tick_params(axis = 'y', labelsize=22)
    ax0.tick_params(axis = 'x', labelsize=22)
    ax0.set_xscale('log')
    ax0.grid(True) 
    # Legend 
    handles0, labels0 = ax0.get_legend_handles_labels()
    fig0.legend(handles0, labels0, loc='lower right', prop={'size': 26}) 
    # Save, show , clean
    if(SAVE):
        fig0.savefig('/tmp/bench_'+names[k]+'_SEED='+str(SEED)+'_MAXITER='+str(MAXITER)+'_TOL='+str(TOL)+'.pdf', bbox_inches="tight")




# # Plot CV
# fig1, ax1 = plt.subplots(1, 1, figsize=(19.2,10.8)) 
# # Create bar plot
# X = np.arange(N_pb)
# b2 = ax1.bar(X, osqp_iter_avg, yerr=osqp_iter_std, color = 'g', width = 0.22, capsize=10, label='FDDP')
# b3 = ax1.bar(X + 0.13, hpipm_ocp_iter_avg, yerr=hpipm_ocp_iter_std, color = 'b', width = 0.22, capsize=10, label='SQP')
# # b1 = ax1.bar(X - 0.13, osqp_iter_avg, yerr=osqp_iter_std, color = 'r', width = 0.25, capsize=10, label='FDDP')
# # b2 = ax1.bar(X + 0.13, hpipm_ocp_iter_avg, yerr=hpipm_ocp_iter_std, color = 'g', width = 0.25, capsize=10, label='SQP')
# # Set axis and stuff
# ax1.set_ylabel('Number of iterations', fontsize=26)
# ax1.set_ylim(-10, MAXITER)
# # ax1.set_yticks(X)
# ax1.tick_params(axis = 'y', labelsize=22)
# # ax1.set_xlabel('Experiment', fontsize=26)
# ax1.set_xticks(X)
# ax1.set_xticklabels(names, rotation='horizontal', fontsize=18)
# ax1.tick_params(axis = 'x', labelsize = 22)
# # ax1.set_title('Performance of SQP and FDDP', fontdict={'size': 26})
# ax1.grid(True) 
# # Legend 
# handles1, labels1 = ax1.get_legend_handles_labels()
# fig1.legend(handles1, labels1, loc='upper right', prop={'size': 26}) #, ncols=2)
# Save, show , clean
# fig0.savefig('/home/skleff/data_paper_CSSQP/bench_'+names[0]+'_SEED='+str(SEED)+'_MAXITER='+str(MAXITER)+'_TOL='+str(TOL)+'.png')


plt.show()
plt.close('all')


# def create_full_qp(problem, ndx, nu, gap, y):
#     '''
#     Creates the full QP (needed for benchmarks)
#     '''
#     P = np.zeros((problem.T*(ndx + nu), problem.T*(ndx + nu)))
#     q = np.zeros(problem.T*(ndx + nu))
#     Asize = problem.T*(ndx + nu)
#     A = np.zeros((problem.T*ndx, Asize))
#     B = np.zeros(problem.T*ndx)
#     for t, (model, data) in enumerate(zip(problem.runningModels, problem.runningDatas)):
#         index_u = problem.T*ndx + t * nu
#         if t>=1:
#             index_x = (t-1) * ndx
#             P[index_x:index_x+ndx, index_x:index_x+ndx] = data.Lxx.copy()
#             P[index_x:index_x+ndx, index_u:index_u+nu] = data.Lxu.copy()
#             P[index_u:index_u+nu, index_x:index_x+ndx] = data.Lxu.T.copy()
#             q[index_x:index_x+ndx] = data.Lx.copy()
#         P[index_u:index_u+nu, index_u:index_u+nu] = data.Luu.copy()
#         q[index_u:index_u+nu] = data.Lu.copy()
#         A[t * ndx: (t+1) * ndx, index_u:index_u+nu] = - data.Fu.copy() 
#         A[t * ndx: (t+1) * ndx, t * ndx: (t+1) * ndx] = np.eye(ndx)
#         if t >=1:
#             A[t * ndx: (t+1) * ndx, (t-1) * ndx: t * ndx] = - data.Fx.copy()
#         B[t * ndx: (t+1) * ndx] = gap[t].copy()
#     P[(problem.T-1)*ndx:problem.T*ndx, problem.T*ndx-ndx:problem.T*ndx] = problem.terminalData.Lxx.copy()
#     q[(problem.T-1)*ndx:problem.T*ndx] = problem.terminalData.Lx.copy()
#     n_in = sum([len(y[i]) for i in range(len(y))])
#     C = np.zeros((n_in, problem.T*(ndx + nu)))
#     l = np.zeros(n_in)
#     u = np.zeros(n_in)
#     nin_count = 0
#     index_x = problem.T*ndx
#     for t, (model, data) in enumerate(zip(problem.runningModels, problem.runningDatas)):
#         if model.ng == 0:
#             continue
#         l[nin_count: nin_count + model.ng] = model.g_lb - data.g
#         u[nin_count: nin_count + model.ng] = model.g_ub - data.g
#         if t > 0:
#             C[nin_count: nin_count + model.ng, (t-1)*ndx: t*ndx] = data.Gx
#         C[nin_count: nin_count + model.ng, index_x+t*nu: index_x+(t+1)*nu] = data.Gu
#         nin_count += model.ng
#     model = problem.terminalModel
#     data = problem.terminalData
#     if model.ng != 0:
#         l[nin_count: nin_count + model.ng] = model.g_lb - data.g
#         u[nin_count: nin_count + model.ng] = model.g_ub - data.g
#         C[nin_count: nin_count + model.ng, (problem.T-1)*ndx: problem.T*ndx] = data.Gx
#         nin_count += model.ng
#     return P, q, A, B, C, u, l
