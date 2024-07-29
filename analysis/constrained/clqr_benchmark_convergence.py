from clqr import ActionModelCLQR
import mim_solvers
import numpy as np
import time
import crocoddyl

# import pathlib
# import os
# python_path = pathlib.Path('/home/ajordana/workspace/mim_solvers/python/').absolute()
# os.sys.path.insert(1, str(python_path))
# from csqp import CSQP

import pathlib
import os
python_path = pathlib.Path('/home/skleff/libs/mim_solvers/python/').absolute()
os.sys.path.insert(1, str(python_path))
from csqp import CSQP

# Solver params
MAXITER     = 1     
TOL         = 1e-4
CALLBACKS   = False
MAX_QP_ITER = 1000
MAX_QP_TIME = int(5e3) # in ms
EPS_ABS     = 1e-1
EPS_REL     = 0.
SAVE        = False # Save figure 

# Benchmark params
SEED = 10 ; np.random.seed(SEED)


names = "clqr"


N_samples = 10


horizon = 40

# dim_list = [20, 40, 60, 80, 100]



# Compute convergence statistics

csqp_time_solved = np.zeros(N_samples)
osqp_time_solved = np.zeros(N_samples)
hpipm_time_solved = np.zeros(N_samples)

nx = 10

# Solvers
SOLVERS = ['CSQP',
           'OSQP',
        #    'HPIPM_DENSE', 
           'HPIPM_OCP']

if('CSQP' in SOLVERS):
    csqp_iter_samples   = []  
    csqp_time_samples   = []
    csqp_solved_samples = []
if('OSQP' in SOLVERS):
    osqp_iter_samples   = []  
    osqp_time_samples   = []
    osqp_solved_samples = []
if('HPIPM_DENSE' in SOLVERS):
    hpipm_dense_iter_samples   = []  
    hpipm_dense_time_samples   = []
    hpipm_dense_solved_samples = []
if('HPIPM_OCP' in SOLVERS):
    hpipm_ocp_iter_samples   = []  
    hpipm_ocp_time_samples   = []
    hpipm_ocp_solved_samples = []


# loop on many problems
for pb_id in range(N_samples):

    runningModels = [ActionModelCLQR(nx, isInitial=True)]
    for j in range(horizon-1):
        runningModels.append(ActionModelCLQR(nx))
    terminalModel = ActionModelCLQR(nx, isTerminal=True)

    x0 = np.zeros(nx)
    problem = crocoddyl.ShootingProblem(
        x0, runningModels, terminalModel
    )


    xs = [np.zeros(nx)] * (horizon+1)
    us = [np.zeros(terminalModel.nu) for _ in range(horizon)]

    # CSQP solver
    print("CSQP")
    solvercsqp = mim_solvers.SolverCSQP(problem)
    solvercsqp.termination_tolerance = TOL
    solvercsqp.max_qp_iters = MAX_QP_ITER
    solvercsqp.eps_abs = EPS_ABS
    solvercsqp.eps_rel = EPS_REL
    solvercsqp.equality_qp_initial_guess = False
    solvercsqp.update_rho_with_heuristic = False
    solvercsqp.with_callbacks = CALLBACKS
    solvercsqp.solve(xs, us, 0)
    t1 = time.time()
    solvercsqp.computeDirection(True)
    solvercsqp.qp_time = time.time() - t1
    solved = (solvercsqp.norm_primal < EPS_ABS and solvercsqp.norm_dual < EPS_ABS and solvercsqp.qp_iters <= MAX_QP_ITER)
    csqp_solved_samples.append( solved )
    if(not solved): 
        print("      FAILED !!!!")
        csqp_iter_samples.append(MAX_QP_ITER)
        csqp_time_samples.append(MAX_QP_TIME)
    else:
        csqp_iter_samples.append(solvercsqp.qp_iters)
        csqp_time_samples.append(solvercsqp.qp_time*1e3)
        print("     QP Time = ", solvercsqp.qp_time)
        print("     QP Iter = ", solvercsqp.qp_iters)
    # csqp_time_solved[pb_id] = csqp_qp_time*1e3

    # OSQP solver
    print("OSQP")
    solverosqp = CSQP(problem, "OSQP")
    solverosqp.termination_tolerance = TOL
    solverosqp.max_qp_iters = MAX_QP_ITER
    solverosqp.eps_abs = EPS_ABS
    solverosqp.eps_rel = EPS_REL
    solverosqp.with_callbacks = CALLBACKS
    solverosqp.solve(xs, us, MAXITER, False)
    solved = (solverosqp.found_qp_sol and solverosqp.norm_primal < EPS_ABS and solverosqp.norm_dual < EPS_ABS and solverosqp.qp_iters <= MAX_QP_ITER)
    osqp_solved_samples.append( solved )
    if(not solved): 
        print("      FAILED !!!!")
        osqp_iter_samples.append(MAX_QP_ITER)
        osqp_time_samples.append(MAX_QP_TIME)
    else:
        osqp_iter_samples.append(solverosqp.qp_iters)
        osqp_time_samples.append(solverosqp.qp_time*1e3)
        print("     QP Time = ", solverosqp.qp_time)
        print("     QP Iter = ", solverosqp.qp_iters)

    # HPIPM OCP
    print("HPIPM_OCP")
    solverhpipm = CSQP(problem, "HPIPM_OCP")
    solverhpipm.termination_tolerance = TOL
    solverhpipm.max_qp_iters = MAX_QP_ITER
    solverhpipm.eps_abs = EPS_ABS
    solverhpipm.eps_rel = EPS_REL
    solverhpipm.with_callbacks = CALLBACKS
    solverhpipm.solve(xs, us, MAXITER, False)
    # Check convergence
    if(solverhpipm.found_qp_sol):
        solved = (solverhpipm.norm_primal < EPS_ABS and solverhpipm.norm_dual < EPS_ABS and solverhpipm.qp_iters <= MAX_QP_ITER)
    else:
        solved = False
    hpipm_ocp_solved_samples.append( solved )
    if(not solved): 
        print("      FAILED !!!!")
        hpipm_ocp_iter_samples.append(MAX_QP_ITER)
        hpipm_ocp_time_samples.append(MAX_QP_TIME)
    else:
        hpipm_ocp_iter_samples.append(solverhpipm.qp_iters)
        hpipm_ocp_time_samples.append(solverhpipm.qp_time*1e3)
    print("     QP Time = ", solverhpipm.qp_time)
    print("     QP Iter = ", solverhpipm.qp_iters)


# Compute convergence statistics
if('CSQP' in SOLVERS):  
    csqp_iter_solved = np.zeros((MAX_QP_ITER, 1))
    csqp_time_solved = np.zeros((MAX_QP_TIME, 1))
if('OSQP' in SOLVERS):  
    osqp_iter_solved = np.zeros((MAX_QP_ITER, 1))
    osqp_time_solved = np.zeros((MAX_QP_TIME, 1))
if('HPIPM_DENSE' in SOLVERS):  
    hpipm_dense_iter_solved = np.zeros((MAX_QP_ITER, 1))
    hpipm_dense_time_solved = np.zeros((MAX_QP_TIME, 1))
if('HPIPM_OCP' in SOLVERS):  
    hpipm_ocp_iter_solved = np.zeros((MAX_QP_ITER, 1))
    hpipm_ocp_time_solved = np.zeros((MAX_QP_TIME, 1))
# Count number of problems solved for each sample initial state 
for i in range(N_samples):
    # For sample i of problem k , compare nb iter to max iter
    if('CSQP' in SOLVERS): 
        csqp_iter_ik  = np.array(csqp_iter_samples)[i]
        csqp_time_ik  = np.array(csqp_time_samples)[i]
    if('OSQP' in SOLVERS): 
        osqp_iter_ik = np.array(osqp_iter_samples)[i]
        osqp_time_ik = np.array(osqp_time_samples)[i]
    if('HPIPM_DENSE' in SOLVERS): 
        hpipm_dense_iter_ik = np.array(hpipm_dense_iter_samples)[i]
        hpipm_dense_time_ik = np.array(hpipm_dense_time_samples)[i]
    if('HPIPM_OCP' in SOLVERS): 
        hpipm_ocp_iter_ik = np.array(hpipm_ocp_iter_samples)[i]
        hpipm_ocp_time_ik = np.array(hpipm_ocp_time_samples)[i]
    # Number of iterations
    for j in range(MAX_QP_ITER):
        if('CSQP' in SOLVERS): 
            if(csqp_iter_ik < j): csqp_iter_solved[j] += 1
        if('OSQP' in SOLVERS): 
            if(osqp_iter_ik < j): osqp_iter_solved[j] += 1
        if('HPIPM_DENSE' in SOLVERS): 
            if(hpipm_dense_iter_ik < j): hpipm_dense_iter_solved[j] += 1
        if('HPIPM_OCP' in SOLVERS): 
            if(hpipm_ocp_iter_ik < j): hpipm_ocp_iter_solved[j] += 1
    # Solve time
    for j in range(MAX_QP_TIME):
        if('CSQP' in SOLVERS): 
            if(csqp_time_ik < j): csqp_time_solved[j] += 1
        if('OSQP' in SOLVERS): 
            if(osqp_time_ik < j): osqp_time_solved[j] += 1
        if('HPIPM_DENSE' in SOLVERS): 
            if(hpipm_dense_time_ik < j): hpipm_dense_time_solved[j] += 1
        if('HPIPM_OCP' in SOLVERS): 
            if(hpipm_ocp_time_ik < j): hpipm_ocp_time_solved[j] += 1




# Generate plot of number of iterations for each problem
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
 
# x-axis : max number of iterations
xdata     = range(0,MAX_QP_ITER)
fig0, ax0 = plt.subplots(1, 1, figsize=(19.2,10.8))
if('CSQP' in SOLVERS): 
    ax0.plot(xdata, csqp_iter_solved[:,0]/N_samples, color='r', linestyle='solid', linewidth=4, label='CSQP') 
if('OSQP' in SOLVERS): 
    ax0.plot(xdata, osqp_iter_solved[:,0]/N_samples, color='y', linestyle='dashed', linewidth=4, label='OSQP') 
if('HPIPM_DENSE' in SOLVERS): 
    ax0.plot(xdata, hpipm_dense_iter_solved[:,0]/N_samples, color='g', linestyle='dotted',linewidth=4, label='HPIPM (dense)') 
if('HPIPM_OCP' in SOLVERS): 
    ax0.plot(xdata, hpipm_ocp_iter_solved[:,0]/N_samples, color='b', linestyle='dashdot', linewidth=4, label='HPIPM (OCP)') #marker='o', markerfacecolor='b', linestyle='-', markersize=12, markeredgecolor='k', alpha=1., label='SQP')
# Set axis and stuff
ax0.set_ylabel('Percentage of problems solved', fontsize=26)
ax0.set_xlabel('Max. number of iterations', fontsize=26)
ax0.set_ylim(-0.02, 1.02)
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
xdata     = range(0,MAX_QP_TIME)
fig0, ax0 = plt.subplots(1, 1, figsize=(19.2,10.8))
if('CSQP' in SOLVERS): 
    ax0.plot(xdata, csqp_time_solved[:,0]/N_samples, color='r', linestyle='solid', linewidth=4, label='CSQP') 
if('OSQP' in SOLVERS): 
    ax0.plot(xdata, osqp_time_solved[:,0]/N_samples, color='y', linestyle='dashed', linewidth=4, label='OSQP') 
if('HPIPM_DENSE' in SOLVERS):
    ax0.plot(xdata, hpipm_dense_time_solved[:,0]/N_samples, color='g', linestyle='dotted', linewidth=4, label='HPIPM (dense)') 
if('HPIPM_OCP' in SOLVERS): 
    ax0.plot(xdata, hpipm_ocp_time_solved[:,0]/N_samples, color='b', linestyle='dashdot', linewidth=4, label='HPIPM (OCP)') #marker='o', markerfacecolor='b', linestyle='-', markersize=12, markeredgecolor='k', alpha=1., label='SQP')
# Set axis and stuff
ax0.set_ylabel('Percentage of problems solved', fontsize=26)
ax0.set_xlabel('Max. solving time (ms)', fontsize=26)
ax0.set_ylim(-0.02, 1.02)
ax0.tick_params(axis = 'y', labelsize=22)
ax0.tick_params(axis = 'x', labelsize=22)
ax0.grid(True) 
# Legend 
handles0, labels0 = ax0.get_legend_handles_labels()
fig0.legend(handles0, labels0, loc='lower right', bbox_to_anchor=(0.902, 0.1), prop={'size': 26}) 
# Save, show , clean
if(SAVE):
    fig0.savefig('/tmp/data_sqp_paper_croc2/bench_'+names[k]+'_SEED='+str(SEED)+'_MAXITER='+str(MAXITER)+'_TOL='+str(TOL)+'.pdf', bbox_inches="tight")


plt.show()
plt.close('all')
