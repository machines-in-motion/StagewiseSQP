from clqr import ActionModelCLQR
import mim_solvers
import numpy as np
import time
import crocoddyl
import pathlib
import os
python_path = pathlib.Path('/home/ajordana/eigen_workspace/mim_solvers/python/').absolute()
os.sys.path.insert(1, str(python_path))
from csqp import CSQP


# Solver params
MAXITER     = 1     
TOL         = 1e-4
CALLBACKS   = False
MAX_QP_ITER = 50
MAX_QP_TIME = int(1e3) # in ms
EPS_ABS     = 1e-100
EPS_REL     = 0.
SAVE        = False # Save figure 

# Benchmark params
SEED = 10 ; np.random.seed(SEED)


names = "clqr"


N_samples = 10


horizon = 10

dim_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# dim_list = [80]



# Compute convergence statistics

csqp_time_solved = np.zeros((len(dim_list), N_samples))
osqp_time_solved = np.zeros((len(dim_list), N_samples))

for k, nx in enumerate(dim_list):
    print("\n dim state = ", nx, "\n")
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
        print(csqp_qp_time)


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
        print(solverosqp.qp_time)

        

csqp_qp_time_mean = np.mean(csqp_time_solved, axis=1)
osqp_qp_time_mean = np.mean(osqp_time_solved, axis=1)


csqp_qp_time_std = np.std(csqp_time_solved, axis=1)
osqp_qp_time_std = np.std(osqp_time_solved, axis=1)



# Generate plot of number of iterations for each problem
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
 

# x-axis : max time allowed to solve the QP (in ms)
xdata     = np.array(dim_list)

fig0, ax0 = plt.subplots(1, 1, figsize=(19.2,10.8))

ax0.plot(xdata, csqp_qp_time_mean, color='r', linestyle='solid', linewidth=4, label='CSQP') 
ax0.plot(xdata, osqp_qp_time_mean, color='y', linestyle='solid', linewidth=4, label='OSQP') 

ax0.fill_between(xdata, csqp_qp_time_mean+csqp_qp_time_std, csqp_qp_time_mean-csqp_qp_time_std, facecolor='r', alpha=0.5)
ax0.fill_between(xdata, osqp_qp_time_mean+osqp_qp_time_std, osqp_qp_time_mean-osqp_qp_time_std, facecolor='y', alpha=0.5)

# Set axis and stuff
ax0.set_ylabel('Time [ms]', fontsize=26)
ax0.set_xlabel('State dimension', fontsize=26)
# ax0.set_ylim(-0.02, 1.02)
ax0.tick_params(axis = 'y', labelsize=22)
ax0.tick_params(axis = 'x', labelsize=22)
ax0.grid(True) 
# Legend 
handles0, labels0 = ax0.get_legend_handles_labels()
fig0.legend(handles0, labels0, loc='lower right', bbox_to_anchor=(0.902, 0.1), prop={'size': 26}) 
# Save, show , clean
if(SAVE):
    fig0.savefig('/tmp/data_sqp_paper_croc2/bench_'+names+'_SEED='+str(SEED)+'_MAXITER='+str(MAXITER)+'_TOL='+str(TOL)+'.pdf', bbox_inches="tight")


plt.show()
plt.close('all')
