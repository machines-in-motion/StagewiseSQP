'''
Compare linear (SQP) vs nonlinear (FDDP) rollouts
For this purpose, filter line-search is used in both solvers
Also compare with FDDP (original LS) and DDP 

- kuka
- quadrotor
- double pendulum
'''
import numpy as np
import crocoddyl
import pinocchio as pin
from problems import create_double_pendulum_problem, create_kuka_problem, create_humanoid_taichi_problem, create_quadrotor_problem
from mim_robots.robot_loader import load_pinocchio_wrapper
import example_robot_data
import mim_solvers
import time
from plot_config import LABELS, COLORS, LINESTYLES, LABELSIZE, FONTSIZE, FIGSIZE

SAVE_PLOT = True

# Benchmark name 
BENCH_NAME = 'Taichi'

if(BENCH_NAME == 'Pendulum'):
    MAXITER = 500 
elif(BENCH_NAME == 'Kuka'):
    MAXITER = 100 
elif(BENCH_NAME == 'Quadrotor'):
    MAXITER = 200 
elif(BENCH_NAME == 'Taichi'):
    MAXITER = 300 
else:
    print("Error: wrong bench name !")

# Solver params
TOL         = 1e-8
CALLBACKS   = True
FILTER_SIZE = MAXITER
SAVE        = True  


# Initial states
pendulum_x0  = np.array([3.14, 0., 0., 0.])
cartpole_x0  = np.array([0., 3.14, 0., 0.])
kuka_x0      = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.] + [0.]*7)
quadrotor    = example_robot_data.load('hector') 
quadrotor_x0 = np.array(list(quadrotor.q0) + [0.]*quadrotor.model.nv) 
humanoid_x0  = np.array([0.4, 0, 1.2])

# Create 1 solver of each type for each problem
print('------')
if(BENCH_NAME == "Pendulum"):  
    pb = create_double_pendulum_problem(pendulum_x0)
if(BENCH_NAME == "Kuka"):      
    pb = create_kuka_problem(kuka_x0) 
if(BENCH_NAME == "Quadrotor"): 
    pb = create_quadrotor_problem(quadrotor_x0) 
if(BENCH_NAME == "Taichi"): 
    pb = create_humanoid_taichi_problem(humanoid_x0) 

# Create solver DDP (SS)
solverddp = mim_solvers.SolverDDP(pb)
solverddp.xs = [solverddp.problem.x0] * (solverddp.problem.T + 1)  
solverddp.us = solverddp.problem.quasiStatic([solverddp.problem.x0] * solverddp.problem.T)
solverddp.termination_tolerance = TOL
if(CALLBACKS): solverddp.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])



# Create solver FDDP (MS)
solverfddp = mim_solvers.SolverFDDP(pb)
solverfddp.xs = [solverfddp.problem.x0] * (solverfddp.problem.T + 1)  
# solverfddp.us = solverfddp.problem.quasiStatic([solverfddp.problem.x0] * solverfddp.problem.T)
solverfddp.termination_tolerance = TOL
solverfddp.use_filter_line_search = False
if(CALLBACKS): solverfddp.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])

# Create solver FDDP_filter (MS)
solverfddp_filter = mim_solvers.SolverFDDP(pb)
solverfddp_filter.xs = [solverfddp_filter.problem.x0] * (solverfddp_filter.problem.T + 1)  
solverfddp_filter.us = solverfddp_filter.problem.quasiStatic([solverfddp_filter.problem.x0] * solverfddp_filter.problem.T)
solverfddp_filter.termination_tolerance  = TOL
solverfddp_filter.use_filter_line_search = True
solverfddp_filter.filter_size            = MAXITER
if(CALLBACKS): solverfddp_filter.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])

# Create solver SQP (MS)
solversqp = mim_solvers.SolverSQP(pb)
solversqp.xs = [solversqp.problem.x0] * (solversqp.problem.T + 1)  
solversqp.us = solversqp.problem.quasiStatic([solversqp.problem.x0] * solversqp.problem.T)
solversqp.termination_tolerance  = TOL
solversqp.use_filter_line_search = True
solversqp.filter_size            = MAXITER
if(CALLBACKS): solversqp.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])




# Initial state
if(BENCH_NAME == "Pendulum"):  x0 = pendulum_x0
if(BENCH_NAME == "Kuka"):      x0 = kuka_x0
if(BENCH_NAME == "Quadrotor"): x0 = quadrotor_x0
if(BENCH_NAME == "Taichi"):    x0 = humanoid_x0

# DDP (SS)
print("   Problem : "+BENCH_NAME+" DDP")
if(BENCH_NAME == 'Taichi'):
    models = list(solverddp.problem.runningModels) + [solverddp.problem.terminalModel]
    for m in models: m.differential.costs.costs["gripperPose"].cost.residual.reference = pin.SE3(np.eye(3), x0.copy())
else:
    solverddp.problem.x0 = x0

solverddp.xs = [solverddp.problem.x0] * (solverddp.problem.T + 1) 
solverddp.us = solverddp.problem.quasiStatic([solverddp.problem.x0] * solverddp.problem.T)
tic = time.time()
solverddp.solve(solverddp.xs, solverddp.us, MAXITER, False)
ddp_solve_time = time.time() - tic
solved = (solverddp.iter < MAXITER) and (solverddp.KKT < TOL)

logddp = solverddp.getCallbacks()[-1]



# FDDP (MS)
print("   Problem : "+BENCH_NAME+" FDDP")
if(BENCH_NAME == 'Taichi'):
    models = list(solverfddp.problem.runningModels) + [solverfddp.problem.terminalModel]
    for m in models: m.differential.costs.costs["gripperPose"].cost.residual.reference = pin.SE3(np.eye(3), x0.copy())
else:
    solverfddp.problem.x0 = x0
solverfddp.xs = [solverfddp.problem.x0] * (solverfddp.problem.T + 1) 
solverfddp.us = solverfddp.problem.quasiStatic([solverfddp.problem.x0] * solverfddp.problem.T)
tic = time.time()
solverfddp.solve(solverfddp.xs, solverfddp.us, MAXITER, False)
fddp_solve_time = time.time() - tic
solved = (solverfddp.iter < MAXITER) and (solverfddp.KKT < TOL)

logfddp = solverfddp.getCallbacks()[-1]

# FDDP filter (MS)
print("   Problem : "+BENCH_NAME+" FDDP_filter")
if(BENCH_NAME == 'Taichi'):
    models = list(solverfddp_filter.problem.runningModels) + [solverfddp_filter.problem.terminalModel]
    for m in models: m.differential.costs.costs["gripperPose"].cost.residual.reference = pin.SE3(np.eye(3), x0.copy())
else:
    solverfddp_filter.problem.x0 = x0
solverfddp_filter.xs = [solverfddp_filter.problem.x0] * (solverfddp_filter.problem.T + 1) 
solverfddp_filter.us = solverfddp_filter.problem.quasiStatic([solverfddp_filter.problem.x0] * solverfddp_filter.problem.T)
tic = time.time()
solverfddp_filter.solve(solverfddp_filter.xs, solverfddp_filter.us, MAXITER, False)
fddp_filter_solve_time = time.time() - tic
solved = (solverfddp_filter.iter < MAXITER) and (solverfddp_filter.KKT < TOL)

logfddp_filter = solverfddp_filter.getCallbacks()[-1]

# SQP        
print("   Problem : "+BENCH_NAME+" SQP")
if(BENCH_NAME == 'Taichi'):
    models = list(solversqp.problem.runningModels) + [solversqp.problem.terminalModel]
    for m in models: m.differential.costs.costs["gripperPose"].cost.residual.reference = pin.SE3(np.eye(3), x0.copy())
else:
    solversqp.problem.x0 = x0
solversqp.xs = [solversqp.problem.x0] * (solversqp.problem.T + 1) 
solversqp.us = solversqp.problem.quasiStatic([solversqp.problem.x0] * solversqp.problem.T)
tic = time.time()
solversqp.solve(solversqp.xs, solversqp.us, MAXITER, False)
sqp_solve_time = time.time() - tic
    # Check convergence
solved = (solversqp.iter < MAXITER) and (solversqp.KKT < TOL)

logsqp = solversqp.getCallbacks()[-1]



fs_ddp = logddp.convergence_data['fs']
fs_fddp = logfddp.convergence_data['fs']
fs_fddp_fiter = logfddp_filter.convergence_data['fs']
fs_sqp = logsqp.convergence_data['fs']

fs_norm_ddp = np.array([np.linalg.norm(np.array(gap_)) for gap_ in fs_ddp])
fs_norm_fddp = np.array([np.linalg.norm(np.array(gap_)) for gap_ in fs_fddp])
fs_norm_fddp_filter = np.array([np.linalg.norm(np.array(gap_)) for gap_ in fs_fddp_fiter])
fs_norm_sqp = np.array([np.linalg.norm(np.array(gap_)) for gap_ in fs_sqp])[1:]


fs_norm_ddp = np.max((fs_norm_ddp, 1e-12 * np.ones(fs_norm_ddp.shape)), axis=0)
fs_norm_fddp = np.max((fs_norm_fddp, 1e-12 * np.ones(fs_norm_fddp.shape)), axis=0)
fs_norm_fddp_filter = np.max((fs_norm_fddp_filter, 1e-12 * np.ones(fs_norm_fddp_filter.shape)), axis=0)
fs_norm_sqp = np.max((fs_norm_sqp, 1e-12 * np.ones(fs_norm_sqp.shape)), axis=0)



cost_ddp = np.array(logddp.convergence_data['cost'])
cost_fddp = np.array(logfddp.convergence_data['cost'])
cost_fddp_filter = np.array(logfddp_filter.convergence_data['cost'])
cost_sqp = np.array(logsqp.convergence_data['cost'])[1:]


cost_ddp -= cost_ddp[-1] + 1e-12
cost_fddp -= cost_fddp[-1] + 1e-12
cost_fddp_filter -= cost_fddp_filter[-1] + 1e-12 
cost_sqp -= cost_sqp[-1] + 1e-12


import numpy as np

from plot_config import LABELS, COLORS, LINESTYLES, LABELSIZE, FONTSIZE, FIGSIZE, LINEWIDTH
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Solvers
SOLVERS = ['DDP',
           'FDDP',
           'FDDP_filter', 
           'SQP']
# Load data 
PREFIX = "data/"
file_name = PREFIX + BENCH_NAME + ".npz"
print("Loading " + file_name)
npzfile = np.load(file_name)
N_SAMPLES = npzfile['N_SAMPLES']
MAXITER   = npzfile['MAXITER']
print("N_SAMPLES = ", N_SAMPLES)
print("MAXITER   = ", MAXITER)


# x-axis : max number of iterations
xdataddp           =  range(1, solverddp.iter + 1)
xdatafddp          =  range(1, solverfddp.iter + 1)
xdatafddp_filter   =  range(1, solverfddp_filter.iter + 1)
xdatasqp           =  range(1, solversqp.iter + 1)
# Plot number of problem solved vs max number of iterations
fig0, (ax0, ax1) = plt.subplots(2, 1, figsize=FIGSIZE)
ax0.plot(xdataddp,         cost_ddp, color=COLORS['DDP'], linewidth=LINEWIDTH, linestyle=LINESTYLES['DDP'], label=LABELS['DDP']) 
ax0.plot(xdatafddp,        cost_fddp, color=COLORS['FDDP'], linewidth=LINEWIDTH, linestyle=LINESTYLES['FDDP'], label=LABELS['FDDP']) 
ax0.plot(xdatafddp_filter, cost_fddp_filter, color=COLORS['FDDP_filter'], linewidth=LINEWIDTH, linestyle=LINESTYLES['FDDP_filter'], label=LABELS['FDDP_filter']) 
ax0.plot(xdatasqp,         cost_sqp, color=COLORS['SQP'], linewidth=LINEWIDTH, linestyle=LINESTYLES['SQP'], label=LABELS['SQP']) 

ax1.plot(xdataddp,         np.array(fs_norm_ddp), color=COLORS['DDP'], linewidth=LINEWIDTH, linestyle=LINESTYLES['DDP'], label=LABELS['DDP']) 
ax1.plot(xdatafddp,        np.array(fs_norm_fddp), color=COLORS['FDDP'], linewidth=LINEWIDTH, linestyle=LINESTYLES['FDDP'], label=LABELS['FDDP']) 
ax1.plot(xdatafddp_filter, np.array(fs_norm_fddp_filter), color=COLORS['FDDP_filter'], linewidth=LINEWIDTH, linestyle=LINESTYLES['FDDP_filter'], label=LABELS['FDDP_filter']) 
ax1.plot(xdatasqp,         np.array(fs_norm_sqp), color=COLORS['SQP'], linewidth=LINEWIDTH, linestyle=LINESTYLES['SQP'], label=LABELS['SQP']) 


# Set axis and stuff
ax0.set_yscale("log")

ax0.set_ylabel('Cost', fontsize=FONTSIZE)
# ax0.set_ylim(-0.02, 1.02)
ax0.tick_params(axis = 'y', labelsize=LABELSIZE)
ax0.tick_params(axis = 'x', labelsize=LABELSIZE)


# Set axis and stuff
ax1.set_yscale("log")
ax1.set_ylabel('Gap norm', fontsize=FONTSIZE)
ax1.set_xlabel('Number of iterations', fontsize=FONTSIZE)
# ax0.set_ylim(-0.02, 1.02)
ax1.tick_params(axis = 'y', labelsize=LABELSIZE)
ax1.tick_params(axis = 'x', labelsize=LABELSIZE)

ax0.grid(True) 
ax1.grid(True) 
# Legend
plt.legend()
# Save, show , clean
if(SAVE_PLOT):
    fig0.savefig('/tmp/bench_'+BENCH_NAME+'.pdf', bbox_inches="tight")

plt.show()
plt.close('all')