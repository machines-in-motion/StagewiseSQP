'''
Compare linear (GNMS) vs nonlinear (FDDP) rollouts
For this purpose, filter line-search is used in both solvers
Also compare with FDDP (original LS) and DDP 

- solo

Randomizing over initial states
'''
import numpy as np
import crocoddyl
import pinocchio as pin


from bench_utils.solo_walking_problem import WholeBodyModel, WholeBodyDDPSolver
import bench_utils.solo_walking_problem as conf

solo_model = WholeBodyModel(conf)

def create_solo_climb_problem(x0):
    ddp_planner = WholeBodyDDPSolver(solo_model, x0,  MPC=False, WARM_START=False)
    return ddp_planner.pb


# Solver params
MAXITER     = 1000 
TOL         = 1e-4 
CALLBACKS   = False
FILTER_SIZE = MAXITER

# Benchmark params
SEED = 1 ; np.random.seed(SEED)
N_samples = 1
names = ['Solo']      # maxiter = 500

N_pb = len(names)

# Solvers
solversDDP         = []
solversFDDP        = []
solversFDDP_filter = []
solversGNMS        = []

# Initial states
solo_x0 = solo_model.x0

# Create 1 solver of each type for each problem
print('------')
for k,name in enumerate(names):
    pb = create_solo_climb_problem(solo_x0)

    # Create solver DDP (SS)
    solverddp = crocoddyl.SolverDDP(pb)
    solverddp.xs = [solverddp.problem.x0] * (solverddp.problem.T + 1)  
    solverddp.us = solverddp.problem.quasiStatic([solverddp.problem.x0] * solverddp.problem.T)
    solverddp.termination_tolerance = TOL
    if(CALLBACKS): solverddp.setCallbacks([crocoddyl.CallbackVerbose()])
    solversDDP.append(solverddp)
    
    # Create solver FDDP (MS)
    solverfddp = crocoddyl.SolverFDDP(pb)
    solverfddp.xs = [solverfddp.problem.x0] * (solverfddp.problem.T + 1)  
    solverfddp.us = solverfddp.problem.quasiStatic([solverfddp.problem.x0] * solverfddp.problem.T)
    solverfddp.termination_tolerance = TOL
    if(CALLBACKS): solverfddp.setCallbacks([crocoddyl.CallbackVerbose()])
    solversFDDP.append(solverfddp)

    # Create solver FDDP_filter (MS)
    solverfddp_filter = crocoddyl.SolverFDDP(pb)
    solverfddp_filter.xs = [solverfddp_filter.problem.x0] * (solverfddp_filter.problem.T + 1)  
    solverfddp_filter.us = solverfddp_filter.problem.quasiStatic([solverfddp_filter.problem.x0] * solverfddp_filter.problem.T)
    solverfddp_filter.termination_tolerance  = TOL
    solverfddp_filter.use_filter_line_search = True
    solverfddp_filter.filter_size            = MAXITER
    if(CALLBACKS): solverfddp_filter.setCallbacks([crocoddyl.CallbackVerbose()])
    solversFDDP_filter.append(solverfddp_filter)

    # Create solver GNMS (MS)
    solvergnms = crocoddyl.SolverGNMS(pb)
    solvergnms.xs = [solvergnms.problem.x0] * (solvergnms.problem.T + 1)  
    solvergnms.us = solvergnms.problem.quasiStatic([solvergnms.problem.x0] * solvergnms.problem.T)
    solvergnms.termination_tol        = TOL
    solvergnms.use_filter_line_search = True
    solvergnms.filter_size            = MAXITER
    solvergnms.with_callbacks         = CALLBACKS
    solversGNMS.append(solvergnms)



# Initial state samples
solo_x0_samples = np.zeros((N_samples, len(solo_model.x0)))
for i in range(N_samples):
    solo_x0_samples[i,:] = solo_x0.copy() #np.concatenate([pin.randomConfiguration(conf.rmodel), np.zeros(conf.rmodel.nv)])

print("Created "+str(N_samples)+" random initial states per model !")

# Solve problems for sample initial states
ddp_iter_samples   = []  
ddp_kkt_samples    = []
ddp_solved_samples = []

fddp_iter_samples   = []  
fddp_kkt_samples    = []
fddp_solved_samples = []

fddp_filter_iter_samples   = []  
fddp_filter_kkt_samples    = []
fddp_filter_solved_samples = []

gnms_iter_samples   = []  
gnms_kkt_samples    = []
gnms_solved_samples = []

for i in range(N_samples):
    ddp_iter_samples.append([])
    ddp_kkt_samples.append([])
    ddp_solved_samples.append([])

    fddp_iter_samples.append([])
    fddp_kkt_samples.append([])
    fddp_solved_samples.append([])

    fddp_filter_iter_samples.append([])
    fddp_filter_kkt_samples.append([])
    fddp_filter_solved_samples.append([])

    gnms_iter_samples.append([])
    gnms_kkt_samples.append([])
    gnms_solved_samples.append([])

    print("---")
    print("Sample "+str(i+1)+'/'+str(N_samples))
    for k,name in enumerate(names):
        # Initial state
        x0 = solo_x0_samples[i,:]

        # DDP (SS)
        print("   Problem : "+name+" DDP")
        solverddp = solversDDP[k]
        solverddp.problem.x0 = x0
        solverddp.xs = [x0] * (solverddp.problem.T + 1) 
        solverddp.us = solverddp.problem.quasiStatic([x0] * solverddp.problem.T)
        solverddp.solve(solverddp.xs, solverddp.us, MAXITER, False)
        solved = (solverddp.iter < MAXITER) and (solverddp.KKT < TOL)
        ddp_solved_samples[i].append( solved )
        print("   iter = "+str(solverddp.iter)+"  |  KKT = "+str(solverddp.KKT))
        if(not solved): 
            print("      FAILED !!!!")
            ddp_iter_samples[i].append(MAXITER)
        else:
            ddp_iter_samples[i].append(solverddp.iter)
        ddp_kkt_samples[i].append(solverddp.KKT)

        # FDDP (MS)
        print("   Problem : "+name+" FDDP")
        solverfddp = solversFDDP[k]
        solverfddp.problem.x0 = x0
        solverfddp.xs = [x0] * (solverfddp.problem.T + 1) 
        solverfddp.us = solverfddp.problem.quasiStatic([x0] * solverfddp.problem.T)
        solverfddp.solve(solverfddp.xs, solverfddp.us, MAXITER, False)
        solved = (solverfddp.iter < MAXITER) and (solverfddp.KKT < TOL)
        fddp_solved_samples[i].append( solved )
        print("   iter = "+str(solverfddp.iter)+"  |  KKT = "+str(solverfddp.KKT))
        if(not solved): 
            print("      FAILED !!!!")
            fddp_iter_samples[i].append(MAXITER)
        else:
            fddp_iter_samples[i].append(solverfddp.iter)
        fddp_kkt_samples[i].append(solverfddp.KKT)

        # FDDP filter (MS)
        print("   Problem : "+name+" FDDP_filter")
        solverfddp_filter = solversFDDP_filter[k]
        solverfddp_filter.problem.x0 = x0
        solverfddp_filter.xs = [x0] * (solverfddp_filter.problem.T + 1) 
        solverfddp_filter.us = solverfddp_filter.problem.quasiStatic([x0] * solverfddp_filter.problem.T)
        solverfddp_filter.solve(solverfddp_filter.xs, solverfddp_filter.us, MAXITER, False)
        solved = (solverfddp_filter.iter < MAXITER) and (solverfddp_filter.KKT < TOL)
        fddp_filter_solved_samples[i].append( solved )
        print("   iter = "+str(solverfddp_filter.iter)+"  |  KKT = "+str(solverfddp_filter.KKT))
        if(not solved): 
            print("      FAILED !!!!")
            fddp_filter_iter_samples[i].append(MAXITER)
        else:
            fddp_filter_iter_samples[i].append(solverfddp_filter.iter)
        fddp_filter_kkt_samples[i].append(solverfddp_filter.KKT)

        # GNMS        
        print("   Problem : "+name+" GNMS")
        solvergnms = solversGNMS[k]
        solvergnms.problem.x0 = x0
        solvergnms.xs = [x0] * (solvergnms.problem.T + 1) 
        solvergnms.us = solvergnms.problem.quasiStatic([x0] * solvergnms.problem.T)
        solvergnms.solve(solvergnms.xs, solvergnms.us, MAXITER, False)
            # Check convergence
        solved = (solvergnms.iter < MAXITER) and (solvergnms.KKT < TOL)
        gnms_solved_samples[i].append( solved )
        print("   iter = "+str(solvergnms.iter)+"  |  KKT = "+str(solvergnms.KKT))
        if(not solved): 
            print("      FAILED !!!!")
            gnms_iter_samples[i].append(MAXITER)
        else:
            gnms_iter_samples[i].append(solvergnms.iter)
        gnms_kkt_samples[i].append(solvergnms.KKT)


# Average fddp iters
ddp_iter_solved = np.zeros((MAXITER, N_pb))
fddp_iter_solved = np.zeros((MAXITER, N_pb))
fddp_filter_iter_solved = np.zeros((MAXITER, N_pb))
gnms_iter_solved = np.zeros((MAXITER, N_pb))

for k,exp in enumerate(names):
    # Count number of problems solved for each sample initial state 
    for i in range(N_samples):
        # For sample i of problem k , compare nb iter to max iter
        ddp_iter_ik  = np.array(ddp_iter_samples)[i,k]
        fddp_iter_ik = np.array(fddp_iter_samples)[i,k]
        fddp_filter_iter_ik = np.array(fddp_filter_iter_samples)[i,k]
        gnms_iter_ik = np.array(gnms_iter_samples)[i,k]
        for j in range(MAXITER):
            if(ddp_iter_ik < j): ddp_iter_solved[j,k] += 1
            if(fddp_iter_ik < j): fddp_iter_solved[j,k] += 1
            if(fddp_filter_iter_ik < j): fddp_filter_iter_solved[j,k] += 1
            if(gnms_iter_ik < j): gnms_iter_solved[j,k] += 1


# Generate plot of number of iterations for each problem
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
 
# x-axis : max number of iterations
xdata     = range(0,MAXITER)
for k in range(N_pb):
    fig0, ax0 = plt.subplots(1, 1, figsize=(19.2,10.8))
    ax0.plot(xdata, ddp_iter_solved[:,k]/N_samples, color='r', linewidth=4, label='DDP') 
    ax0.plot(xdata, fddp_iter_solved[:,k]/N_samples, color='y', linewidth=4, label='FDDP (default LS)') 
    ax0.plot(xdata, fddp_filter_iter_solved[:,k]/N_samples, color='g', linewidth=4, label='FDDP (filter LS)') 
    ax0.plot(xdata, gnms_iter_solved[:,k]/N_samples, color='b', linewidth=4, label='SQP') #marker='o', markerfacecolor='b', linestyle='-', markersize=12, markeredgecolor='k', alpha=1., label='GNMS')
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
    fig0.savefig('/home/skleff/data_paper_fadmm/bench_'+names[k]+'_SEED='+str(SEED)+'_MAXITER='+str(MAXITER)+'_TOL='+str(TOL)+'.pdf', bbox_inches="tight")


plt.show()
plt.close('all')