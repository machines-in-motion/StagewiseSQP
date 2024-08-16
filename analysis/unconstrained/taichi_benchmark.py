'''
Compare linear (SQP) vs nonlinear (FDDP) rollouts
For this purpose, filter line-search is used in both solvers
Also compare with FDDP (original LS) and DDP 

- humanoid taichi

Randomizing over end-effector goal position
'''

import sys
import numpy as np
import crocoddyl
import pinocchio as pin

import example_robot_data
import mim_solvers
import time

from problems import create_humanoid_taichi_problem

# Solver params
MAXITER     = 300 
TOL         = 1e-4
CALLBACKS   = False
FILTER_SIZE = MAXITER
SAVE        = True # Save figure 

# Benchmark params
SEED = 1 ; np.random.seed(SEED)
N_samples = 2
names = ['Humanoid']

N_pb = len(names)

# Solvers
solversDDP         = []
solversFDDP        = []
solversFDDP_filter = []
solverCSSQP        = []
humanoid_x0  = np.array([0.4, 0, 1.2])

print('------')
for k,name in enumerate(names):
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
    solverfddp_filter.termination_tolerance = TOL
    solverfddp_filter.use_filter_line_search = True
    solverfddp_filter.filter_size = MAXITER
    if(CALLBACKS): solverfddp_filter.setCallbacks([crocoddyl.CallbackVerbose()])
    solversFDDP_filter.append(solverfddp_filter)

    # Create solver SQP (MS)
    solverSQP = mim_solvers.SolverSQP(pb)
    solverSQP.xs = [solverSQP.problem.x0] * (solverSQP.problem.T + 1)  
    solverSQP.us = solverSQP.problem.quasiStatic([solverSQP.problem.x0] * solverSQP.problem.T)
    solverSQP.termination_tolerance = TOL
    solverSQP.with_callbacks = CALLBACKS
    solverSQP.use_filter_line_search = True
    solverSQP.filter_size = MAXITER
    solverCSSQP.append(solverSQP)


# Initial state samples
humanoid_x0_samples  = np.zeros((N_samples, 3))
for i in range(N_samples):
    err = np.zeros(3)
    err[2] = 2*np.random.rand(1) - 1
    humanoid_x0_samples[i,:]  = np.array([0.4, 0, 1.2]) + 0.5*err

print("Created "+str(N_samples)+" random initial states per model !")

# Solve problems for sample initial states
ddp_iter_samples = []  
ddp_kkt_samples  =  []
ddp_solved_samples  =  []
ddp_avg_time_per_iter_samples = []

fddp_iter_samples = []  
fddp_kkt_samples  =  []
fddp_solved_samples  =  []
fddp_avg_time_per_iter_samples = []

fddp_filter_iter_samples = []  
fddp_filter_kkt_samples  =  []
fddp_filter_solved_samples  =  []
fddp_filter_avg_time_per_iter_samples = []

SQP_iter_samples = []  
SQP_kkt_samples  =  []
SQP_solved_samples  =  []
SQP_avg_time_per_iter_samples = []

for i in range(N_samples):
    ddp_iter_samples.append([])
    ddp_kkt_samples.append([])
    ddp_solved_samples.append([])
    ddp_avg_time_per_iter_samples.append([])

    fddp_iter_samples.append([])
    fddp_kkt_samples.append([])
    fddp_solved_samples.append([])
    fddp_avg_time_per_iter_samples.append([])

    fddp_filter_iter_samples.append([])
    fddp_filter_kkt_samples.append([])
    fddp_filter_solved_samples.append([])
    fddp_filter_avg_time_per_iter_samples.append([])

    SQP_iter_samples.append([])
    SQP_kkt_samples.append([])
    SQP_solved_samples.append([])
    SQP_avg_time_per_iter_samples.append([])

    print("---")
    print("Sample "+str(i+1)+'/'+str(N_samples))
    for k,name in enumerate(names):
        # Initial state
        # if(name == "Humanoid"):  
        x0 = humanoid_x0_samples[i,:].copy()
        # DDP (SS)
        print("   Problem : "+name+" DDP")
        solverddp = solversDDP[k]
        models = list(solverddp.problem.runningModels) + [solverddp.problem.terminalModel]
        for m in models: m.differential.costs.costs["gripperPose"].cost.residual.reference = pin.SE3(np.eye(3), x0.copy())
        solverddp.xs = [solverddp.problem.x0] * (solverddp.problem.T + 1) 
        solverddp.us = solverddp.problem.quasiStatic([solverddp.problem.x0] * solverddp.problem.T)
        tic = time.time()
        solverddp.solve(solverddp.xs, solverddp.us, MAXITER, False)
        ddp_solve_time = time.time() - tic
            # Check convergence
        solved = (solverddp.iter < MAXITER) and (solverddp.KKT < TOL)
        ddp_solved_samples[i].append( solved )
        print("   iter = "+str(solverddp.iter)+"  |  KKT = "+str(solverddp.KKT))
        if(not solved): 
            print("      FAILED !!!!")
            ddp_iter_samples[i].append(MAXITER)
        else:
            ddp_iter_samples[i].append(solverddp.iter)
        ddp_avg_time_per_iter_samples[i].append(ddp_solve_time/solverddp.iter)
        ddp_kkt_samples[i].append(solverddp.KKT)

        # FDDP (MS)
        print("   Problem : "+name+" FDDP")
        solverfddp = solversFDDP[k]
        models = list(solverfddp.problem.runningModels) + [solverfddp.problem.terminalModel]
        for m in models: m.differential.costs.costs["gripperPose"].cost.residual.reference = pin.SE3(np.eye(3), x0.copy())
        solverfddp.xs = [solverfddp.problem.x0] * (solverfddp.problem.T + 1) 
        assert(solverfddp.use_filter_line_search == False)
        solverfddp.us = solverfddp.problem.quasiStatic([solverfddp.problem.x0] * solverfddp.problem.T)
        tic = time.time()
        solverfddp.solve(solverfddp.xs, solverfddp.us, MAXITER, False)
        fddp_solve_time = time.time() - tic
            # Check convergence
        solved = (solverfddp.iter < MAXITER) and (solverfddp.KKT < TOL)
        fddp_solved_samples[i].append( solved )
        print("   iter = "+str(solverfddp.iter)+"  |  KKT = "+str(solverfddp.KKT))
        if(not solved): 
            print("      FAILED !!!!")
            fddp_iter_samples[i].append(MAXITER)
        else:
            fddp_iter_samples[i].append(solverfddp.iter)
        fddp_avg_time_per_iter_samples[i].append(fddp_solve_time/solverfddp.iter)
        fddp_kkt_samples[i].append(solverfddp.KKT)

        # FDDP filter (MS)
        print("   Problem : "+name+" FDDP_filter")
        solverfddp_filter = solversFDDP_filter[k]
        models = list(solverfddp_filter.problem.runningModels) + [solverfddp_filter.problem.terminalModel]
        for m in models: m.differential.costs.costs["gripperPose"].cost.residual.reference = pin.SE3(np.eye(3), x0.copy())
        solverfddp_filter.xs = [solverfddp_filter.problem.x0] * (solverfddp_filter.problem.T + 1) 
        solverfddp_filter.us = solverfddp_filter.problem.quasiStatic([solverfddp_filter.problem.x0] * solverfddp_filter.problem.T)
        assert(solverfddp_filter.use_filter_line_search == True)
        assert(solverfddp_filter.filter_size == MAXITER)
        tic = time.time()
        solverfddp_filter.solve(solverfddp_filter.xs, solverfddp_filter.us, MAXITER, False)
        fddp_filter_solve_time = time.time() - tic
            # Check convergence
        solved = (solverfddp_filter.iter < MAXITER) and (solverfddp_filter.KKT < TOL)
        fddp_filter_solved_samples[i].append( solved )
        print("   iter = "+str(solverfddp_filter.iter)+"  |  KKT = "+str(solverfddp_filter.KKT))
        if(not solved): 
            print("      FAILED !!!!")
            fddp_filter_iter_samples[i].append(MAXITER)
        else:
            fddp_filter_iter_samples[i].append(solverfddp_filter.iter)
        fddp_filter_avg_time_per_iter_samples[i].append(fddp_filter_solve_time/solverfddp_filter.iter)
        fddp_filter_kkt_samples[i].append(solverfddp_filter.KKT)

        #  SQP        
        print("   Problem : "+name+" SQP")
        solverSQP = solverCSSQP[k]
        models = list(solverSQP.problem.runningModels) + [solverSQP.problem.terminalModel]
        for m in models: m.differential.costs.costs["gripperPose"].cost.residual.reference = pin.SE3(np.eye(3), x0.copy())
        solverSQP.xs = [solverSQP.problem.x0] * (solverSQP.problem.T + 1) 
        solverSQP.us = solverSQP.problem.quasiStatic([solverSQP.problem.x0] * solverSQP.problem.T)
        tic = time.time()
        solverSQP.solve(solverSQP.xs, solverSQP.us, MAXITER, False)
        SQP_solve_time = time.time() - tic
            # Check convergence
        solved = (solverSQP.iter < MAXITER) and (solverSQP.KKT < TOL)
        SQP_solved_samples[i].append( solved )
        print("   iter = "+str(solverSQP.iter)+"  |  KKT = "+str(solverSQP.KKT))
        if(not solved): 
            print("      FAILED !!!!")
            SQP_iter_samples[i].append(MAXITER)
        else:
            SQP_iter_samples[i].append(solverSQP.iter)
        SQP_avg_time_per_iter_samples[i].append(SQP_solve_time/solverSQP.iter)
        SQP_kkt_samples[i].append(solverSQP.KKT)


# Average fddp iters

ddp_iter_solved = np.zeros((MAXITER, N_pb))
fddp_iter_solved = np.zeros((MAXITER, N_pb))
fddp_filter_iter_solved = np.zeros((MAXITER, N_pb))
SQP_iter_solved = np.zeros((MAXITER, N_pb))

for k,exp in enumerate(names):
    # Count number of problems solved for each sample initial state 
    for i in range(N_samples):
        # For sample i of problem k , compare nb iter to max iter
        ddp_iter_ik  = np.array(ddp_iter_samples)[i,k]
        fddp_iter_ik = np.array(fddp_iter_samples)[i,k]
        fddp_filter_iter_ik = np.array(fddp_filter_iter_samples)[i,k]
        SQP_iter_ik = np.array(SQP_iter_samples)[i,k]
        for j in range(MAXITER):
            if(ddp_iter_ik < j): ddp_iter_solved[j,k] += 1
            if(fddp_iter_ik < j): fddp_iter_solved[j,k] += 1
            if(fddp_filter_iter_ik < j): fddp_filter_iter_solved[j,k] += 1
            if(SQP_iter_ik < j): SQP_iter_solved[j,k] += 1

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

import time
file_name = "/tmp/Taichi_data_"+str(time.ctime())
np.savez_compressed(file_name, 
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

# Generate plot of number of iterations for each problem
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# x-axis : max number of iterations
xdata     = range(0,MAXITER)
xdata2     = range(0,N_samples)

for k in range(N_pb):
    fig0, ax0 = plt.subplots(1, 1, figsize=(19.2,10.8))
    ax0.plot(xdata, ddp_iter_solved[:,k]/N_samples, color='r', linestyle='dashdot', linewidth=4, label='DDP') 
    ax0.plot(xdata, fddp_iter_solved[:,k]/N_samples, color='y', linewidth=4, linestyle='dashed', label='FDDP (default LS)') 
    ax0.plot(xdata, fddp_filter_iter_solved[:,k]/N_samples, color='g', linewidth=4, linestyle='dotted', label='FDDP (filter LS)') 
    ax0.plot(xdata, SQP_iter_solved[:,k]/N_samples, color='b', linewidth=4, linestyle='solid', label='SQP') #marker='o', markerfacecolor='b', linestyle='-', markersize=12, markeredgecolor='k', alpha=1., label='SQP')
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
        fig0.savefig('/tmp/bench_Taichi_SEED='+str(SEED)+'_MAXITER='+str(MAXITER)+'_TOL='+str(TOL)+'_MIM_SOLVERS_v1.pdf', bbox_inches="tight")

        # fig1, ax1 = plt.subplots(1, 1, figsize=(19.2,10.8))
        # ax1.plot(xdata2, ddp_avg_time_per_iter_samples, color='r', linewidth=4, label='DDP') 
        # ax1.plot(xdata2, np.mean(ddp_avg_time_per_iter_samples)*np.ones(N_samples), color='r', linewidth=8, alpha=0.3, linestyle='-.') 
        # ax1.plot(xdata2, fddp_avg_time_per_iter_samples, color='y', linewidth=4, label='FDDP (default LS)') 
        # ax1.plot(xdata2, np.mean(fddp_avg_time_per_iter_samples)*np.ones(N_samples), color='y', linewidth=8, alpha=0.3, linestyle='-.') 
        # ax1.plot(xdata2, fddp_filter_avg_time_per_iter_samples, color='g', linewidth=4, label='FDDP (filter LS)') 
        # ax1.plot(xdata2, np.mean(fddp_filter_avg_time_per_iter_samples)*np.ones(N_samples), color='g', linewidth=8, alpha=0.3, linestyle='-.') 
        # ax1.plot(xdata2, SQP_avg_time_per_iter_samples, color='b', linewidth=4, label='SQP') #marker='o', markerfacecolor='b', linestyle='-', markersize=12, markeredgecolor='k', alpha=1., label='SQP')
        # ax1.plot(xdata2, np.mean(SQP_avg_time_per_iter_samples)*np.ones(N_samples), color='b', linewidth=8, alpha=0.3, linestyle='-.') #marker='o', markerfacecolor='b', linestyle='-', markersize=12, markeredgecolor='k', alpha=1., label='SQP')
        # # Set axis and stuff
        # ax1.set_ylabel('Avg time per iteration', fontsize=26)
        # ax1.set_xlabel('Random problem', fontsize=26)
        # # ax1.set_ylim(-0.02, 1.02)
        # ax1.tick_params(axis = 'y', labelsize=22)
        # ax1.tick_params(axis = 'x', labelsize=22)
        # ax1.grid(True) 
        # # Legend 
        # handles1, labels1 = ax1.get_legend_handles_labels()
        # fig1.legend(handles1, labels1, loc='lower right', bbox_to_anchor=(0.902, 0.1), prop={'size': 26}) 
        # # Save, show , clean
        # if(SAVE):
        #     fig1.savefig('/home/skleff/Desktop/TRO-SQP/data/rollout_benchmarks/bench_Taichi_SEED='+str(SEED)+'_MAXITER='+str(MAXITER)+'_TOL='+str(TOL)+'AVG_TIME.pdf', bbox_inches="tight")


plt.show()
plt.close('all')

