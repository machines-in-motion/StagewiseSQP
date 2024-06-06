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

Randomizing over initial states
'''
import numpy as np
import crocoddyl
import pinocchio as pin

from robot_properties_kuka.config import IiwaConfig
import example_robot_data
# from unconstrained.bench_utils.cartpole_swingup import DifferentialActionModelCartpole
from crocoddyl.utils.pendulum import CostModelDoublePendulum, ActuationModelDoublePendulum
import mim_solvers

import pathlib
import os
python_path = pathlib.Path('/home/skleff/libs/mim_solvers/python/').absolute()
os.sys.path.insert(1, str(python_path))
from csqp import CSQP


def create_double_pendulum_problem(x0):
    '''
    Create shooting problem for the double pendulum model
    '''
    print("Created double pendulum problem ...")
    # Loading the double pendulum model
    pendulum = example_robot_data.load('double_pendulum')
    model = pendulum.model
    state = crocoddyl.StateMultibody(model)
    actuation = ActuationModelDoublePendulum(state, actLink=1)
    nu = actuation.nu
    runningCostModel = crocoddyl.CostModelSum(state, nu)
    terminalCostModel = crocoddyl.CostModelSum(state, nu)
    xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
    xActivation = crocoddyl.ActivationModelQuad(state.ndx)
    uResidual = crocoddyl.ResidualModelControl(state, nu)
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    xPendCost = CostModelDoublePendulum(state, crocoddyl.ActivationModelWeightedQuad(np.array([1.] * 4 + [0.1] * 2)), nu)
    dt = 1e-2
    runningCostModel.addCost("uReg", uRegCost, 1e-4 / dt)
    runningCostModel.addCost("xGoal", xPendCost, 1e-5 / dt)
    terminalCostModel.addCost("xGoal", xPendCost, 100.)
    runningModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel), dt)
    terminalModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel), dt)
    T = 100
    pb = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
    return pb

def create_cartpole_problem(x0):
    '''
    Create shooting problem for Cartpole
    '''
    print("Create cartpole problem ...")
    # Creating the DAM for the cartpole
    cartpoleDAM = DifferentialActionModelCartpole()
    # Using NumDiff for computing the derivatives. We specify the
    # withGaussApprox=True to have approximation of the Hessian based on the
    # Jacobian of the cost residuals.
    cartpoleND = crocoddyl.DifferentialActionModelNumDiff(cartpoleDAM, True)
    # Getting the IAM using the simpletic Euler rule
    timeStep = 5e-2
    cartpoleIAM = crocoddyl.IntegratedActionModelEuler(cartpoleND, timeStep)
    # Creating the shooting problem
    T = 50
    terminalCartpole = DifferentialActionModelCartpole()
    terminalCartpoleDAM = crocoddyl.DifferentialActionModelNumDiff(terminalCartpole, True)
    terminalCartpoleIAM = crocoddyl.IntegratedActionModelEuler(terminalCartpoleDAM, 0.)
    terminalCartpole.costWeights[0] = 200
    terminalCartpole.costWeights[1] = 200
    terminalCartpole.costWeights[2] = 1.
    terminalCartpole.costWeights[3] = 0.1
    terminalCartpole.costWeights[4] = 0.01
    terminalCartpole.costWeights[5] = 0.0001
    pb = crocoddyl.ShootingProblem(x0, [cartpoleIAM] * T, terminalCartpoleIAM)
    return pb 

def create_kuka_problem(x0):
    '''
    Create shooting problem for kuka reaching task
    '''
    print("Create kuka problem ...")
    iiwa_config = IiwaConfig()
    robot       = iiwa_config.buildRobotWrapper()
    model = robot.model
    nq = model.nq; nv = model.nv
    # State and actuation model
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFull(state)
    # Running and terminal cost models
    runningCostModel = crocoddyl.CostModelSum(state)
    terminalCostModel = crocoddyl.CostModelSum(state)
    # Create cost terms 
    # Control regularization cost
    uResidual = crocoddyl.ResidualModelControlGrav(state)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    # State regularization cost
    xResidual = crocoddyl.ResidualModelState(state, x0)
    xRegCost = crocoddyl.CostModelResidual(state, xResidual)
    # endeff frame translation cost
    endeff_frame_id = model.getFrameId("contact")
    # endeff_translation = robot.data.oMf[endeff_frame_id].translation.copy()
    endeff_translation = np.array([0.7, 0, 1.1]) # move endeff +30 cm along x in WORLD frame
    frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, endeff_frame_id, endeff_translation)
    frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)
    # Add costs
    runningCostModel.addCost("stateReg", xRegCost, 1e-1)
    runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
    runningCostModel.addCost("translation", frameTranslationCost, 1)
    terminalCostModel.addCost("stateReg", xRegCost, 1e-1)
    terminalCostModel.addCost("translation", frameTranslationCost, 1)
    # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
    running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel)
    terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel)
    # Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
    dt = 1e-2
    runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
    terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)
    # Create the shooting problem
    T = 50
    pb = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
    return pb

def create_quadrotor_problem(x0):
    '''
    Create shooting problem for quadrotor task
    '''
    print("Create quadrotor problem ...")
    hector = example_robot_data.load('hector')
    robot_model = hector.model
    target_pos = np.array([1., 0., 1.])
    target_quat = pin.Quaternion(1., 0., 0., 0.)
    state = crocoddyl.StateMultibody(robot_model)
    d_cog, cf, cm, u_lim, l_lim = 0.1525, 6.6e-5, 1e-6, 5., 0.1
    tau_f = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.], [1., 1., 1., 1.], [0., d_cog, 0., -d_cog],
                    [-d_cog, 0., d_cog, 0.], [-cm / cf, cm / cf, -cm / cf, cm / cf]])
    actuation = crocoddyl.ActuationModelMultiCopterBase(state, tau_f)

    nu = actuation.nu
    runningCostModel = crocoddyl.CostModelSum(state, nu)
    terminalCostModel = crocoddyl.CostModelSum(state, nu)
    # Costs
    xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
    xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([0.1] * 3 + [1000.] * 3 + [1000.] * robot_model.nv))
    uResidual = crocoddyl.ResidualModelControl(state, nu)
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    goalTrackingResidual = crocoddyl.ResidualModelFramePlacement(state, robot_model.getFrameId("base_link"),
                                                                pin.SE3(target_quat.matrix(), target_pos), nu)
    goalTrackingCost = crocoddyl.CostModelResidual(state, goalTrackingResidual)
    runningCostModel.addCost("xReg", xRegCost, 1e-6)
    runningCostModel.addCost("uReg", uRegCost, 1e-6)
    runningCostModel.addCost("trackPose", goalTrackingCost, 1e-2)
    terminalCostModel.addCost("goalPose", goalTrackingCost, 3.)
    dt = 3e-2
    runningModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel), dt)
    terminalModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel), dt)

    # Creating the shooting problem and the FDDP solver
    T = 33
    pb = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
    return pb 




# Solver params
MAXITER     = 100 
TOL         = 1e-4 
CALLBACKS   = False
MAX_QP_ITER = 25
EPS_ABS     = 1e-20
EPS_REL     = 0.
SAVE        = False # Save figure 

# Benchmark params
SEED = 1 ; np.random.seed(SEED)
N_samples = 100
names = [
    #    'Pendulum'] # maxiter = 500
         'Kuka'] # maxiter = 100
        # #  'Cartpole']  #--> need to explain why it doesn't converge otherwise leave it out 
        #  'Quadrotor'] # maxiter = 200

N_pb = len(names)

# Solvers
solversCSQP        = [] # mim_solvers.SolverCSQP(problem)
solversOSQP        = [] # CSQP(problem, "StagewiseQP")
solversHPIPM_dense = [] # CSQP(problem, "OSQP")
solversHPIPM_ocp   = [] # CSQP(problem, "HPIPM")

# Initial states
pendulum_x0  = np.array([3.14, 0., 0., 0.])
cartpole_x0  = np.array([0., 3.14, 0., 0.])
kuka_x0      = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.] + [0.]*7)
quadrotor    = example_robot_data.load('hector') 
quadrotor_x0 = np.array(list(quadrotor.q0) + [0.]*quadrotor.model.nv) 

# Create 1 solver of each type for each problem
print('------')
for k,name in enumerate(names):
    if(name == "Pendulum"):  
        pb = create_double_pendulum_problem(pendulum_x0)
    if(name == "Cartpole"):  
        pb = create_cartpole_problem(cartpole_x0) 
    if(name == "Kuka"):      
        pb = create_kuka_problem(kuka_x0) 
    if(name == "Quadrotor"): 
        pb = create_quadrotor_problem(quadrotor_x0) 

    # Create solver CSQP 
    solvercsqp = mim_solvers.SolverCSQP(pb)
    solvercsqp.xs = [solvercsqp.problem.x0] * (solvercsqp.problem.T + 1)  
    solvercsqp.us = solvercsqp.problem.quasiStatic([solvercsqp.problem.x0] * solvercsqp.problem.T)
    solvercsqp.termination_tolerance = TOL
    solvercsqp.max_qp_iters = MAX_QP_ITER
    solvercsqp.eps_abs = EPS_ABS
    solvercsqp.eps_rel = EPS_REL
    solvercsqp.equality_qp_initial_guess = False
    solvercsqp.update_rho_with_heuristic = True
    if(CALLBACKS): solvercsqp.setCallbacks([crocoddyl.CallbackVerbose()])
    solversCSQP.append(solvercsqp)
    
    # Create solver OSQP
    solverosqp = CSQP(pb, "OSQP")
    solverosqp.xs = [solverosqp.problem.x0] * (solverosqp.problem.T + 1)  
    solverosqp.us = solverosqp.problem.quasiStatic([solverosqp.problem.x0] * solverosqp.problem.T)
    solverosqp.termination_tolerance = TOL
    solverosqp.max_qp_iters = MAX_QP_ITER
    solverosqp.eps_abs = EPS_ABS
    solverosqp.eps_rel = EPS_REL
    solverosqp.equality_qp_initial_guess = False
    if(CALLBACKS): solverosqp.setCallbacks([crocoddyl.CallbackVerbose()])
    solversOSQP.append(solverosqp)

    # Create solver HPIPM dense
    solverhpipm_dense = CSQP(pb, "HPIPM")
    solverhpipm_dense.xs = [solverhpipm_dense.problem.x0] * (solverhpipm_dense.problem.T + 1)  
    solverhpipm_dense.us = solverhpipm_dense.problem.quasiStatic([solverhpipm_dense.problem.x0] * solverhpipm_dense.problem.T)
    solverhpipm_dense.termination_tolerance  = TOL
    solverhpipm_dense.max_qp_iters = MAX_QP_ITER
    solverhpipm_dense.eps_abs = EPS_ABS
    solverhpipm_dense.eps_rel = EPS_REL
    solverhpipm_dense.equality_qp_initial_guess = False

    if(CALLBACKS): solverhpipm_dense.setCallbacks([crocoddyl.CallbackVerbose()])
    solversHPIPM_dense.append(solverhpipm_dense)

    # Create solver HPIPM ocp
    solverhpipm_ocp = CSQP(pb, "HPIPM")
    solverhpipm_ocp.xs = [solverhpipm_ocp.problem.x0] * (solverhpipm_ocp.problem.T + 1)  
    solverhpipm_ocp.us = solverhpipm_ocp.problem.quasiStatic([solverhpipm_ocp.problem.x0] * solverhpipm_ocp.problem.T)
    solverhpipm_ocp.termination_tolerance  = TOL
    solverhpipm_ocp.with_callbacks         = CALLBACKS
    solverhpipm_ocp.eps_abs = EPS_ABS
    solverhpipm_ocp.eps_rel = EPS_REL
    solverhpipm_ocp.equality_qp_initial_guess = False
    solversHPIPM_ocp.append(solverhpipm_ocp)


# Initial state samples
pendulum_x0_samples  = np.zeros((N_samples, 4))
cartpole_x0_samples  = np.zeros((N_samples, 4))
iiwa_config          = IiwaConfig()
kuka                 = iiwa_config.buildRobotWrapper()
kuka_x0_samples      = np.zeros((N_samples, kuka.model.nq + kuka.model.nv))
quadrotor            = example_robot_data.load('hector') 
humanoid             = example_robot_data.load('talos')
quadrotor_x0_samples = np.zeros((N_samples, quadrotor.model.nq + quadrotor.model.nv))
for i in range(N_samples):
    pendulum_x0_samples[i,:]  = np.array([np.pi*(2*np.random.rand()-1), 0., 0., 0.])
    cartpole_x0_samples[i,:]  = np.array([0., np.pi/2, 0., 0.])
    kuka_x0_samples[i,:]      = np.concatenate([pin.randomConfiguration(kuka.model), np.zeros(kuka.model.nv)])
    quadrotor_x0_samples[i,:] = np.concatenate([pin.randomConfiguration(quadrotor.model), np.zeros(quadrotor.model.nv)])

print("Created "+str(N_samples)+" random initial states per model !")

# Solve problems for sample initial states
csqp_iter_samples   = []  
csqp_kkt_samples    = []
csqp_solved_samples = []

osqp_iter_samples   = []  
osqp_kkt_samples    = []
osqp_solved_samples = []

hpipm_dense_iter_samples   = []  
hpipm_dense_kkt_samples    = []
hpipm_dense_solved_samples = []

hpipm_ocp_iter_samples   = []  
hpipm_ocp_kkt_samples    = []
hpipm_ocp_solved_samples = []

for i in range(N_samples):
    csqp_iter_samples.append([])
    csqp_kkt_samples.append([])
    csqp_solved_samples.append([])

    osqp_iter_samples.append([])
    osqp_kkt_samples.append([])
    osqp_solved_samples.append([])

    hpipm_dense_iter_samples.append([])
    hpipm_dense_kkt_samples.append([])
    hpipm_dense_solved_samples.append([])

    hpipm_ocp_iter_samples.append([])
    hpipm_ocp_kkt_samples.append([])
    hpipm_ocp_solved_samples.append([])

    print("---")
    print("Sample "+str(i+1)+'/'+str(N_samples))
    for k,name in enumerate(names):
        # Initial state
        if(name == "Pendulum"):  x0 = pendulum_x0_samples[i,:]
        if(name == "Cartpole"):  x0 = cartpole_x0_samples[i,:]
        if(name == "Kuka"):      x0 = kuka_x0_samples[i,:]
        if(name == "Quadrotor"): x0 = quadrotor_x0_samples[i,:]

        # DDP (SS)
        print("   Problem : "+name+" CSQP")
        solvercsqp = solversCSQP[k]
        solvercsqp.problem.x0 = x0
        solvercsqp.xs = [x0] * (solvercsqp.problem.T + 1) 
        solvercsqp.us = solvercsqp.problem.quasiStatic([x0] * solvercsqp.problem.T)
        solvercsqp.solve(solvercsqp.xs, solvercsqp.us, MAXITER, False)
        solved = (solvercsqp.iter < MAXITER) and (solvercsqp.KKT < TOL)
        csqp_solved_samples[i].append( solved )
        print("   iter = "+str(solvercsqp.iter)+"  |  KKT = "+str(solvercsqp.KKT))
        if(not solved): 
            print("      FAILED !!!!")
            csqp_iter_samples[i].append(MAXITER)
        else:
            csqp_iter_samples[i].append(solvercsqp.iter)
        csqp_kkt_samples[i].append(solvercsqp.KKT)

        # FDDP (MS)
        print("   Problem : "+name+" OSQP")
        solverosqp = solversOSQP[k]
        solverosqp.problem.x0 = x0
        solverosqp.xs = [x0] * (solverosqp.problem.T + 1) 
        solverosqp.us = solverosqp.problem.quasiStatic([x0] * solverosqp.problem.T)
        solverosqp.solve(solverosqp.xs, solverosqp.us, MAXITER, False)
        solved = (solverosqp.iter < MAXITER) and (solverosqp.KKT < TOL)
        osqp_solved_samples[i].append( solved )
        print("   iter = "+str(solverosqp.iter)+"  |  KKT = "+str(solverosqp.KKT))
        if(not solved): 
            print("      FAILED !!!!")
            osqp_iter_samples[i].append(MAXITER)
        else:
            osqp_iter_samples[i].append(solverosqp.iter)
        osqp_kkt_samples[i].append(solverosqp.KKT)

        # FDDP filter (MS)
        print("   Problem : "+name+" HPIPM_DENSE")
        solverhpipm_dense = solversHPIPM_dense[k]
        solverhpipm_dense.problem.x0 = x0
        solverhpipm_dense.xs = [x0] * (solverhpipm_dense.problem.T + 1) 
        solverhpipm_dense.us = solverhpipm_dense.problem.quasiStatic([x0] * solverhpipm_dense.problem.T)
        solverhpipm_dense.solve(solverhpipm_dense.xs, solverhpipm_dense.us, MAXITER, False)
        solved = (solverhpipm_dense.iter < MAXITER) and (solverhpipm_dense.KKT < TOL)
        hpipm_dense_solved_samples[i].append( solved )
        print("   iter = "+str(solverhpipm_dense.iter)+"  |  KKT = "+str(solverhpipm_dense.KKT))
        if(not solved): 
            print("      FAILED !!!!")
            hpipm_dense_iter_samples[i].append(MAXITER)
        else:
            hpipm_dense_iter_samples[i].append(solverhpipm_dense.iter)
        hpipm_dense_kkt_samples[i].append(solverhpipm_dense.KKT)

        # SQP        
        print("   Problem : "+name+" HPIPM_OCP")
        solverhpipm_ocp = solversHPIPM_ocp[k]
        solverhpipm_ocp.problem.x0 = x0
        solverhpipm_ocp.xs = [x0] * (solverhpipm_ocp.problem.T + 1) 
        solverhpipm_ocp.us = solverhpipm_ocp.problem.quasiStatic([x0] * solverhpipm_ocp.problem.T)
        solverhpipm_ocp.solve(solverhpipm_ocp.xs, solverhpipm_ocp.us, MAXITER, False)
            # Check convergence
        solved = (solverhpipm_ocp.iter < MAXITER) and (solverhpipm_ocp.KKT < TOL)
        hpipm_ocp_solved_samples[i].append( solved )
        print("   iter = "+str(solverhpipm_ocp.iter)+"  |  KKT = "+str(solverhpipm_ocp.KKT))
        if(not solved): 
            print("      FAILED !!!!")
            hpipm_ocp_iter_samples[i].append(MAXITER)
        else:
            hpipm_ocp_iter_samples[i].append(solverhpipm_ocp.iter)
        hpipm_ocp_kkt_samples[i].append(solverhpipm_ocp.KKT)


# Average fddp iters
csqp_iter_solved = np.zeros((MAXITER, N_pb))
osqp_iter_solved = np.zeros((MAXITER, N_pb))
hpipm_dense_iter_solved = np.zeros((MAXITER, N_pb))
hpipm_ocp_iter_solved = np.zeros((MAXITER, N_pb))

for k,exp in enumerate(names):
    # Count number of problems solved for each sample initial state 
    for i in range(N_samples):
        # For sample i of problem k , compare nb iter to max iter
        csqp_iter_ik  = np.array(csqp_iter_samples)[i,k]
        osqp_iter_ik = np.array(osqp_iter_samples)[i,k]
        hpipm_dense_iter_ik = np.array(hpipm_dense_iter_samples)[i,k]
        hpipm_ocp_iter_ik = np.array(hpipm_ocp_iter_samples)[i,k]
        for j in range(MAXITER):
            if(csqp_iter_ik < j): csqp_iter_solved[j,k] += 1
            if(osqp_iter_ik < j): osqp_iter_solved[j,k] += 1
            if(hpipm_dense_iter_ik < j): hpipm_dense_iter_solved[j,k] += 1
            if(hpipm_ocp_iter_ik < j): hpipm_ocp_iter_solved[j,k] += 1


# Generate plot of number of iterations for each problem
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
 
# x-axis : max number of iterations
xdata     = range(0,MAXITER)
for k in range(N_pb):
    fig0, ax0 = plt.subplots(1, 1, figsize=(19.2,10.8))
    ax0.plot(xdata, csqp_iter_solved[:,k]/N_samples, color='r', linewidth=4, label='CSQP') 
    ax0.plot(xdata, osqp_iter_solved[:,k]/N_samples, color='y', linewidth=4, label='OSQP') 
    ax0.plot(xdata, hpipm_dense_iter_solved[:,k]/N_samples, color='g', linewidth=4, label='HPIPM (dense)') 
    ax0.plot(xdata, hpipm_ocp_iter_solved[:,k]/N_samples, color='b', linewidth=4, label='HPIPM (OCP)') #marker='o', markerfacecolor='b', linestyle='-', markersize=12, markeredgecolor='k', alpha=1., label='SQP')
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
    # # Save, show , clean
    # if(SAVE):
    #     fig0.savefig('/home/skleff/data_sqp_paper_croc2/bench_'+names[k]+'_SEED='+str(SEED)+'_MAXITER='+str(MAXITER)+'_TOL='+str(TOL)+'.pdf', bbox_inches="tight")


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