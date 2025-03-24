from clqr import ActionModelCLQR
import mim_solvers
import numpy as np
import time
import crocoddyl
import pathlib
import os
# python_path = pathlib.Path('/home/ajordana/eigen_workspace/mim_solvers/python/').absolute()
python_path = pathlib.Path('/home/skleff/workspace_native/mim_solvers/python/').absolute()
os.sys.path.insert(1, str(python_path))
from csqp import CSQP

from plot_config import LABELS, COLORS, LINESTYLES

# Solver params
MAXITER     = 1     
TOL         = 1e-4
CALLBACKS   = False
MAX_QP_ITER = 25
MAX_QP_TIME = int(1e3) # in ms
EPS_ABS     = 1e-100
EPS_REL     = 0.
SAVE        = True # Save figure 

# Benchmark params
SEED = 10 ; np.random.seed(SEED)

# Solvers
SOLVERS = ['CSQP',
           'OSQP']
        #    'HPIPM_DENSE', 
        #    'HPIPM_OCP']

names = "clqr"

N_samples = 100
nx = 50
dim_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# dim_list = [10, 30, 50, 70, 90] #, 80, 90, 100]



# Compute convergence statistics

csqp_time_solved = np.zeros((len(dim_list), N_samples))
osqp_time_solved = np.zeros((len(dim_list), N_samples))
hpipm_dense_time_solved = np.zeros((len(dim_list), N_samples))
hpipm_ocp_time_solved = np.zeros((len(dim_list), N_samples))

for k, horizon in enumerate(dim_list):
    print("\n dim horizon = ", horizon, "\n")
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
        if('CSQP' in SOLVERS):
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
            csqp_qp_time = time.time() - t1
            csqp_time_solved[k, pb_id] = csqp_qp_time*1e3
            print("csqp = ", csqp_qp_time)
            print("csqp = ", solvercsqp.qp_iters)

        if('OSQP' in SOLVERS):
            solverosqp = CSQP(problem, "OSQP")
            solverosqp.xs = [solverosqp.problem.x0] * (solverosqp.problem.T + 1)  
            solverosqp.us = solverosqp.problem.quasiStatic([solverosqp.problem.x0] * solverosqp.problem.T)
            solverosqp.termination_tolerance = TOL
            solverosqp.max_qp_iters = MAX_QP_ITER
            solverosqp.eps_abs = EPS_ABS
            solverosqp.eps_rel = EPS_REL
            solverosqp.with_callbacks = CALLBACKS
            solverosqp.solve(solverosqp.xs, solverosqp.us, MAXITER, False)
            osqp_time_solved[k, pb_id] = solverosqp.qp_time*1e3
            print("osqp = ", solverosqp.qp_time)
            print("osqp = ", solverosqp.qp_iters)

        if('HPIPM_DENSE' in SOLVERS):
            hpipm_dense = CSQP(problem, "HPIPM_DENSE")
            hpipm_dense.xs = [hpipm_dense.problem.x0] * (hpipm_dense.problem.T + 1)  
            hpipm_dense.us = hpipm_dense.problem.quasiStatic([hpipm_dense.problem.x0] * hpipm_dense.problem.T)
            hpipm_dense.termination_tolerance = TOL
            hpipm_dense.max_qp_iters = MAX_QP_ITER
            hpipm_dense.eps_abs = EPS_ABS
            hpipm_dense.eps_rel = EPS_REL
            hpipm_dense.with_callbacks = CALLBACKS
            hpipm_dense.solve(hpipm_dense.xs, hpipm_dense.us, MAXITER, False)
            hpipm_dense_time_solved[k, pb_id] = hpipm_dense.qp_time*1e3
            print("hpipm_dense = ", hpipm_dense.qp_time)
            print("hpipm_dense = ", hpipm_dense.qp_iters)

        if('HPIPM_OCP' in SOLVERS):
            hpipm_ocp = CSQP(problem, "HPIPM_OCP")
            hpipm_ocp.xs = [hpipm_ocp.problem.x0] * (hpipm_ocp.problem.T + 1)  
            hpipm_ocp.us = hpipm_ocp.problem.quasiStatic([hpipm_ocp.problem.x0] * hpipm_ocp.problem.T)
            hpipm_ocp.termination_tolerance = TOL
            hpipm_ocp.max_qp_iters = MAX_QP_ITER
            hpipm_ocp.eps_abs = EPS_ABS
            hpipm_ocp.eps_rel = EPS_REL
            hpipm_ocp.with_callbacks = CALLBACKS
            hpipm_ocp.solve(hpipm_ocp.xs, hpipm_ocp.us, MAXITER, False)
            hpipm_ocp_time_solved[k, pb_id] = hpipm_ocp.qp_time*1e3
            print("hpipm_ocp = ", hpipm_ocp.qp_time)
            print("hpipm_ocp = ", hpipm_ocp.qp_iters)

csqp_qp_time_mean        = np.mean(csqp_time_solved, axis=1)
osqp_qp_time_mean        = np.mean(osqp_time_solved, axis=1)
hpipm_dense_qp_time_mean = np.mean(hpipm_dense_time_solved, axis=1)
hpipm_ocp_qp_time_mean   = np.mean(hpipm_ocp_time_solved, axis=1)

csqp_qp_time_std        = np.std(csqp_time_solved, axis=1)
osqp_qp_time_std        = np.std(osqp_time_solved, axis=1)
hpipm_dense_qp_time_std = np.std(hpipm_dense_time_solved, axis=1)
hpipm_ocp_qp_time_std   = np.std(hpipm_ocp_time_solved, axis=1)

# Save data
if(SAVE):
    PREFIX = 'data/'
    file_name = PREFIX + "CLQR_horizon_benchmark"
    np.savez(file_name, 
             N_samples=N_samples,
             nx=nx,
             dim_list=dim_list,
             csqp_qp_time_mean=csqp_qp_time_mean, 
             osqp_qp_time_mean=osqp_qp_time_mean,
             hpipm_dense_qp_time_mean=hpipm_dense_qp_time_mean,
             hpipm_ocp_qp_time_mean=hpipm_ocp_qp_time_mean, 
             csqp_qp_time_std=csqp_qp_time_std,
             osqp_qp_time_std=osqp_qp_time_std,
             hpipm_dense_qp_time_std=hpipm_dense_qp_time_std,
             hpipm_ocp_qp_time_std=hpipm_ocp_qp_time_std)
    PREFIX = '/home/skleff/SQP_REBUTAL_BENCH/'
    file_name = PREFIX + "CLQR_horizon_benchmark"
    np.savez(file_name, 
             N_samples=N_samples,
             nx=nx,
             dim_list=dim_list,
             csqp_qp_time_mean=csqp_qp_time_mean, 
             osqp_qp_time_mean=osqp_qp_time_mean,
             hpipm_dense_qp_time_mean=hpipm_dense_qp_time_mean,
             hpipm_ocp_qp_time_mean=hpipm_ocp_qp_time_mean, 
             csqp_qp_time_std=csqp_qp_time_std,
             osqp_qp_time_std=osqp_qp_time_std,
             hpipm_dense_qp_time_std=hpipm_dense_qp_time_std,
             hpipm_ocp_qp_time_std=hpipm_ocp_qp_time_std)