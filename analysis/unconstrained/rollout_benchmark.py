'''
Compare linear (SQP) vs nonlinear (FDDP) rollouts
For this purpose, filter line-search is used in both solvers
Also compare with FDDP (original LS) and DDP 

- kuka
- quadrotor
- double pendulum

Randomizing over initial states
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

# Benchmark name 
BENCH_NAME = 'Taichi'
# BENCH_NAME must be :
#  'Pendulum'  # maxiter = 500
#  'Kuka'      # maxiter = 100
#  'Quadrotor' # maxiter = 200
#  'Taichi'    # maxiter = 300
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
TOL         = 1e-4 
CALLBACKS   = False
FILTER_SIZE = MAXITER
SAVE        = True  

# Benchmark params
SEED = 1 ; np.random.seed(SEED)
N_SAMPLES = 100

# Solvers
solversDDP         = []
solversFDDP        = []
solversFDDP_filter = []
solverCSSQP        = []

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
if(CALLBACKS): solverddp.setCallbacks([crocoddyl.CallbackVerbose()])
solversDDP.append(solverddp)

# Create solver FDDP (MS)
solverfddp = mim_solvers.SolverFDDP(pb)
solverfddp.xs = [solverfddp.problem.x0] * (solverfddp.problem.T + 1)  
solverfddp.us = solverfddp.problem.quasiStatic([solverfddp.problem.x0] * solverfddp.problem.T)
solverfddp.termination_tolerance = TOL
solverfddp.use_filter_line_search = False
if(CALLBACKS): solverfddp.setCallbacks([crocoddyl.CallbackVerbose()])
solversFDDP.append(solverfddp)

# Create solver FDDP_filter (MS)
solverfddp_filter = mim_solvers.SolverFDDP(pb)
solverfddp_filter.xs = [solverfddp_filter.problem.x0] * (solverfddp_filter.problem.T + 1)  
solverfddp_filter.us = solverfddp_filter.problem.quasiStatic([solverfddp_filter.problem.x0] * solverfddp_filter.problem.T)
solverfddp_filter.termination_tolerance  = TOL
solverfddp_filter.use_filter_line_search = True
solverfddp_filter.filter_size            = MAXITER
if(CALLBACKS): solverfddp_filter.setCallbacks([crocoddyl.CallbackVerbose()])
solversFDDP_filter.append(solverfddp_filter)

# Create solver SQP (MS)
solverSQP = mim_solvers.SolverSQP(pb)
solverSQP.xs = [solverSQP.problem.x0] * (solverSQP.problem.T + 1)  
solverSQP.us = solverSQP.problem.quasiStatic([solverSQP.problem.x0] * solverSQP.problem.T)
solverSQP.termination_tolerance  = TOL
solverSQP.use_filter_line_search = True
solverSQP.filter_size            = MAXITER
solverSQP.with_callbacks         = CALLBACKS
solverCSSQP.append(solverSQP)


# Initial state samples
pendulum_x0_samples  = np.zeros((N_SAMPLES, 4))
cartpole_x0_samples  = np.zeros((N_SAMPLES, 4))
kuka                 = load_pinocchio_wrapper("iiwa")
kuka_x0_samples      = np.zeros((N_SAMPLES, kuka.model.nq + kuka.model.nv))
quadrotor            = example_robot_data.load('hector') 
humanoid             = example_robot_data.load('talos')
quadrotor_x0_samples = np.zeros((N_SAMPLES, quadrotor.model.nq + quadrotor.model.nv))
humanoid_x0_samples  = np.zeros((N_SAMPLES, 3))
for i in range(N_SAMPLES):
    pendulum_x0_samples[i,:]  = np.array([np.pi*(2*np.random.rand()-1), 0., 0., 0.])
    cartpole_x0_samples[i,:]  = np.array([0., np.pi/2, 0., 0.])
    kuka_x0_samples[i,:]      = np.concatenate([pin.randomConfiguration(kuka.model), np.zeros(kuka.model.nv)])
    quadrotor_x0_samples[i,:] = np.concatenate([pin.randomConfiguration(quadrotor.model), np.zeros(quadrotor.model.nv)])
    err = np.zeros(3)
    err[2] = 2*np.random.rand(1) - 1
    humanoid_x0_samples[i,:]  = np.array([0.4, 0, 1.2]) + 0.5*err

print("Created "+str(N_SAMPLES)+" random initial states per model !")

# Solve problems for sample initial states
ddp_iter_samples   = []  
ddp_kkt_samples    = []
ddp_solved_samples = []
ddp_avg_time_per_iter_samples = []

fddp_iter_samples   = []  
fddp_kkt_samples    = []
fddp_solved_samples = []
fddp_avg_time_per_iter_samples = []

fddp_filter_iter_samples   = []  
fddp_filter_kkt_samples    = []
fddp_filter_solved_samples = []
fddp_filter_avg_time_per_iter_samples = []

SQP_iter_samples   = []  
SQP_kkt_samples    = []
SQP_solved_samples = []
SQP_avg_time_per_iter_samples = []

# Solve the problem for each sample
for i in range(N_SAMPLES):
    print("---")
    print("Sample "+str(i+1)+'/'+str(N_SAMPLES))
    # Initial state
    if(BENCH_NAME == "Pendulum"):  x0 = pendulum_x0_samples[i,:]
    if(BENCH_NAME == "Kuka"):      x0 = kuka_x0_samples[i,:]
    if(BENCH_NAME == "Quadrotor"): x0 = quadrotor_x0_samples[i,:]
    if(BENCH_NAME == "Taichi"):    x0 = humanoid_x0_samples[i,:]

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
    ddp_solved_samples.append( solved )
    print("   iter = "+str(solverddp.iter)+"  |  KKT = "+str(solverddp.KKT))
    if(not solved): 
        print("      FAILED !!!!")
        ddp_iter_samples.append(MAXITER)
    else:
        ddp_iter_samples.append(solverddp.iter)
    ddp_avg_time_per_iter_samples.append(ddp_solve_time/solverddp.iter)
    ddp_kkt_samples.append(solverddp.KKT)

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
    fddp_solved_samples.append( solved )
    print("   iter = "+str(solverfddp.iter)+"  |  KKT = "+str(solverfddp.KKT))
    if(not solved): 
        print("      FAILED !!!!")
        fddp_iter_samples.append(MAXITER)
    else:
        fddp_iter_samples.append(solverfddp.iter)
    fddp_avg_time_per_iter_samples.append(fddp_solve_time/solverfddp.iter)
    fddp_kkt_samples.append(solverfddp.KKT)

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
    fddp_filter_solved_samples.append( solved )
    print("   iter = "+str(solverfddp_filter.iter)+"  |  KKT = "+str(solverfddp_filter.KKT))
    if(not solved): 
        print("      FAILED !!!!")
        fddp_filter_iter_samples.append(MAXITER)
    else:
        fddp_filter_iter_samples.append(solverfddp_filter.iter)
    fddp_filter_avg_time_per_iter_samples.append(fddp_filter_solve_time/solverfddp_filter.iter)
    fddp_filter_kkt_samples.append(solverfddp_filter.KKT)

    # SQP        
    print("   Problem : "+BENCH_NAME+" SQP")
    if(BENCH_NAME == 'Taichi'):
        models = list(solverSQP.problem.runningModels) + [solverSQP.problem.terminalModel]
        for m in models: m.differential.costs.costs["gripperPose"].cost.residual.reference = pin.SE3(np.eye(3), x0.copy())
    else:
        solverSQP.problem.x0 = x0
    solverSQP.xs = [solverSQP.problem.x0] * (solverSQP.problem.T + 1) 
    solverSQP.us = solverSQP.problem.quasiStatic([solverSQP.problem.x0] * solverSQP.problem.T)
    tic = time.time()
    solverSQP.solve(solverSQP.xs, solverSQP.us, MAXITER, False)
    SQP_solve_time = time.time() - tic
        # Check convergence
    solved = (solverSQP.iter < MAXITER) and (solverSQP.KKT < TOL)
    SQP_solved_samples.append( solved )
    print("   iter = "+str(solverSQP.iter)+"  |  KKT = "+str(solverSQP.KKT))
    if(not solved): 
        print("      FAILED !!!!")
        SQP_iter_samples.append(MAXITER)
    else:
        SQP_iter_samples.append(solverSQP.iter)
    SQP_avg_time_per_iter_samples.append(SQP_solve_time/solverSQP.iter)
    SQP_kkt_samples.append(solverSQP.KKT)

# Compute solving time statistics
ddp_mean_solve_time         = np.mean(np.array(ddp_avg_time_per_iter_samples))
ddp_std_solve_time          = np.std(np.array(ddp_avg_time_per_iter_samples))
fddp_mean_solve_time        = np.mean(np.array(fddp_avg_time_per_iter_samples))
fddp_std_solve_time         = np.std(np.array(fddp_avg_time_per_iter_samples))
fddp_filter_mean_solve_time = np.mean(np.array(fddp_filter_avg_time_per_iter_samples))
fddp_filter_std_solve_time  = np.std(np.array(fddp_filter_avg_time_per_iter_samples))
SQP_mean_solve_time         = np.mean(np.array(SQP_avg_time_per_iter_samples))
SQP_std_solve_time          = np.std(np.array(SQP_avg_time_per_iter_samples))

print("Average solving times \n")
print(ddp_mean_solve_time)
print(" DDP      = " , ddp_mean_solve_time         ,  ' \xB1 ' , ddp_std_solve_time)
print(" FDDP     = " , fddp_mean_solve_time        ,  ' \xB1 ' , fddp_std_solve_time)
print(" FDDP_LS  = " , fddp_filter_mean_solve_time ,  ' \xB1 ' , fddp_filter_std_solve_time)
print(" SQP      = " , SQP_mean_solve_time         ,  ' \xB1 ' , SQP_std_solve_time)

# Average fddp iters
ddp_iter_solved         = np.zeros(MAXITER)
fddp_iter_solved        = np.zeros(MAXITER)
fddp_filter_iter_solved = np.zeros(MAXITER)
SQP_iter_solved         = np.zeros(MAXITER)

# Count number of problems solved for each sample initial state 
for i in range(N_SAMPLES):
    # For sample i of problem k , compare nb iter to max iter
    ddp_iter_ik  = np.array(ddp_iter_samples)[i]
    fddp_iter_ik = np.array(fddp_iter_samples)[i]
    fddp_filter_iter_ik = np.array(fddp_filter_iter_samples)[i]
    SQP_iter_ik = np.array(SQP_iter_samples)[i]
    for j in range(MAXITER):
        if(ddp_iter_ik < j): ddp_iter_solved[j] += 1
        if(fddp_iter_ik < j): fddp_iter_solved[j] += 1
        if(fddp_filter_iter_ik < j): fddp_filter_iter_solved[j] += 1
        if(SQP_iter_ik < j): SQP_iter_solved[j] += 1

# Save the benchmark data 
file_name = "/home/skleff/SQP_REBUTAL_BENCH/"+BENCH_NAME 
np.savez_compressed(file_name, 
         N_SAMPLES=N_SAMPLES,
         MAXITER=MAXITER,
         ddp_iter_solved=ddp_iter_solved, 
         fddp_iter_solved=fddp_iter_solved,
         fddp_filter_iter_solved=fddp_filter_iter_solved,
         SQP_iter_solved=SQP_iter_solved, 
         ddp_mean_solve_time=ddp_mean_solve_time,
         ddp_std_solve_time=ddp_std_solve_time,
         fddp_mean_solve_time=fddp_mean_solve_time, 
         fddp_std_solve_time=fddp_std_solve_time,
         fddp_filter_mean_solve_time=fddp_filter_mean_solve_time,
         fddp_filter_std_solve_time=fddp_filter_std_solve_time, 
         SQP_mean_solve_time=SQP_mean_solve_time,
         SQP_std_solve_time=SQP_std_solve_time)