import sys
import numpy as np
import crocoddyl
import pinocchio as pin

from robot_properties_kuka.config import IiwaConfig
import example_robot_data
from cartpole_swingup import DifferentialActionModelCartpole
from crocoddyl.utils.pendulum import CostModelDoublePendulum, ActuationModelDoublePendulum

def create_double_pendulum_problem(x0, solver_name='FDDP'):
    '''
    Create shooting problem for the double pendulum model
    '''
    print("Created double pendulum problem ("+solver_name+")...")
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
    # Warm-start 
    xs0 = [pb.x0] * (pb.T + 1)
    us0 = pb.quasiStatic([pb.x0] * pb.T)
    
    # Solver
    if(solver_name == 'GNMS'):
        solver = crocoddyl.SolverGNMS(pb)
    else:
        solver = crocoddyl.SolverFDDP(pb)
    
    solver.xs = xs0
    solver.us = us0

    return solver #, xs0, us0

def create_cartpole_problem(x0, solver_name='FDDP'):
    '''
    Create shooting problem for Cartpole
    '''
    print("Create cartpole problem ("+solver_name+")...")
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
    # Warm-start 
    xs0 = [pb.x0] * (pb.T + 1)
    us0 = pb.quasiStatic([pb.x0] * pb.T)
    
    # Solver
    if(solver_name == 'GNMS'):
        solver = crocoddyl.SolverGNMS(pb)
    else:
        solver = crocoddyl.SolverFDDP(pb)
    
    solver.xs = xs0
    solver.us = us0

    return solver #, xs0, us0

def create_kuka_problem(x0, solver_name='FDDP'):
    '''
    Create shooting problem for kuka reaching task
    '''
    print("Create kuka problem ("+solver_name+")...")
    robot = IiwaConfig.buildRobotWrapper()
    model = robot.model
    nq = model.nq; nv = model.nv; nu = nq; nx = nq+nv
    q0 = x0[:nq] 
    v0 = x0[nq:] #v0 = np.zeros(nv)
    # x0 = np.concatenate([q0, v0]).copy()
    robot.framesForwardKinematics(q0)
    robot.computeJointJacobians(q0)
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
    # Warm-start 
    xs0 = [pb.x0] * (pb.T + 1)
    us0 = pb.quasiStatic([pb.x0] * pb.T)
    
    # Solver
    if(solver_name == 'GNMS'):
        solver = crocoddyl.SolverGNMS(pb)
    else:
        solver = crocoddyl.SolverFDDP(pb)
    
    solver.xs = xs0
    solver.us = us0

    return solver #, xs0, us0

def create_quadrotor_problem(x0, solver_name='FDDP'):
    '''
    Create shooting problem for quadrotor task
    '''
    print("Create quadrotor problem ("+solver_name+")...")
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
    # x0 = np.concatenate([hector.q0, np.zeros(state.nv)])
    pb = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
    
    # Warm-start 
    xs0 = [pb.x0] * (pb.T + 1)
    us0 = pb.quasiStatic([pb.x0] * pb.T)
    
    # Solver
    if(solver_name == 'GNMS'):
        solver = crocoddyl.SolverGNMS(pb)
    else:
        solver = crocoddyl.SolverFDDP(pb)
    
    solver.xs = xs0
    solver.us = us0

    return solver #, xs0, us0

def create_humanoid_taichi_problem(x0, solver_name='FDDP'):
    '''
    Create shooting problem for Talos taichi task
    '''
    print("Create humanoid problem ("+solver_name+")...")
    # Load robot
    robot = example_robot_data.load('talos')
    rmodel = robot.model
    # Create data structures
    rdata = rmodel.createData()
    state = crocoddyl.StateMultibody(rmodel)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    # Set integration time
    DT = 5e-2
    T = 40
    target = np.array([0.4, 0, 1.2])
    # Initialize reference state, target and reference CoM
    rightFoot = 'right_sole_link'
    leftFoot = 'left_sole_link'
    endEffector = 'gripper_left_joint'
    endEffectorId = rmodel.getFrameId(endEffector)
    rightFootId = rmodel.getFrameId(rightFoot)
    leftFootId = rmodel.getFrameId(leftFoot)
    q0 = rmodel.referenceConfigurations["half_sitting"]
    x0 = np.concatenate([q0, np.zeros(rmodel.nv)])
    pin.forwardKinematics(rmodel, rdata, q0)
    pin.updateFramePlacements(rmodel, rdata)
    rfPos0 = rdata.oMf[rightFootId].translation
    lfPos0 = rdata.oMf[leftFootId].translation
    refGripper = rdata.oMf[rmodel.getFrameId("gripper_left_joint")].translation
    comRef = (rfPos0 + lfPos0) / 2
    comRef[2] = pin.centerOfMass(rmodel, rdata, q0)[2].item()
    # Create two contact models used along the motion
    contactModel1Foot = crocoddyl.ContactModelMultiple(state, actuation.nu)
    contactModel2Feet = crocoddyl.ContactModelMultiple(state, actuation.nu)
    supportContactModelLeft = crocoddyl.ContactModel6D(state, leftFootId, pin.SE3.Identity(), actuation.nu,
                                                    np.array([0, 40]))
    supportContactModelRight = crocoddyl.ContactModel6D(state, rightFootId, pin.SE3.Identity(), actuation.nu,
                                                        np.array([0, 40]))
    contactModel1Foot.addContact(rightFoot + "_contact", supportContactModelRight)
    contactModel2Feet.addContact(leftFoot + "_contact", supportContactModelLeft)
    contactModel2Feet.addContact(rightFoot + "_contact", supportContactModelRight)
    # Cost for self-collision
    maxfloat = sys.float_info.max
    xlb = np.concatenate([ 
        -maxfloat * np.ones(6),  # dimension of the SE(3) manifold
        rmodel.lowerPositionLimit[7:],
        -maxfloat * np.ones(state.nv)
    ])
    xub = np.concatenate([
        maxfloat * np.ones(6),  # dimension of the SE(3) manifold
        rmodel.upperPositionLimit[7:],
        maxfloat * np.ones(state.nv)
    ])
    bounds = crocoddyl.ActivationBounds(xlb, xub, 1.)
    xLimitResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
    xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
    limitCost = crocoddyl.CostModelResidual(state, xLimitActivation, xLimitResidual)
    # Cost for state and control
    xResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
    xActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([0] * 3 + [10.] * 3 + [0.01] * (state.nv - 6) + [10] * state.nv)**2)
    uResidual = crocoddyl.ResidualModelControl(state, actuation.nu)
    xTActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([0] * 3 + [10.] * 3 + [0.01] * (state.nv - 6) + [100] * state.nv)**2)
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    xRegTermCost = crocoddyl.CostModelResidual(state, xTActivation, xResidual)
    # Cost for target reaching: hand and foot
    handTrackingResidual = crocoddyl.ResidualModelFramePlacement(state, endEffectorId, pin.SE3(np.eye(3), target),
                                                                actuation.nu)
    handTrackingActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1] * 3 + [0.0001] * 3)**2)
    handTrackingCost = crocoddyl.CostModelResidual(state, handTrackingActivation, handTrackingResidual)

    footTrackingResidual = crocoddyl.ResidualModelFramePlacement(state, leftFootId,
                                                                pin.SE3(np.eye(3), np.array([0., 0.4, 0.])),
                                                                actuation.nu)
    footTrackingActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1, 1, 0.1] + [1.] * 3)**2)
    footTrackingCost1 = crocoddyl.CostModelResidual(state, footTrackingActivation, footTrackingResidual)
    footTrackingResidual = crocoddyl.ResidualModelFramePlacement(state, leftFootId,
                                                                pin.SE3(np.eye(3), np.array([0.3, 0.15, 0.35])),
                                                                actuation.nu)
    footTrackingCost2 = crocoddyl.CostModelResidual(state, footTrackingActivation, footTrackingResidual)
    # Cost for CoM reference
    comResidual = crocoddyl.ResidualModelCoMPosition(state, comRef, actuation.nu)
    comTrack = crocoddyl.CostModelResidual(state, comResidual)
    # Create cost model per each action model. We divide the motion in 3 phases plus its terminal model
    runningCostModel1 = crocoddyl.CostModelSum(state, actuation.nu)
    runningCostModel2 = crocoddyl.CostModelSum(state, actuation.nu)
    runningCostModel3 = crocoddyl.CostModelSum(state, actuation.nu)
    terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)
    # Then let's add the running and terminal cost functions
    runningCostModel1.addCost("gripperPose", handTrackingCost, 1e2)
    runningCostModel1.addCost("stateReg", xRegCost, 1e-3)
    runningCostModel1.addCost("ctrlReg", uRegCost, 1e-4)
    runningCostModel1.addCost("limitCost", limitCost, 1e3)

    runningCostModel2.addCost("gripperPose", handTrackingCost, 1e2)
    runningCostModel2.addCost("footPose", footTrackingCost1, 1e1)
    runningCostModel2.addCost("stateReg", xRegCost, 1e-3)
    runningCostModel2.addCost("ctrlReg", uRegCost, 1e-4)
    runningCostModel2.addCost("limitCost", limitCost, 1e3)

    runningCostModel3.addCost("gripperPose", handTrackingCost, 1e2)
    runningCostModel3.addCost("footPose", footTrackingCost2, 1e1)
    runningCostModel3.addCost("stateReg", xRegCost, 1e-3)
    runningCostModel3.addCost("ctrlReg", uRegCost, 1e-4)
    runningCostModel3.addCost("limitCost", limitCost, 1e3)

    terminalCostModel.addCost("gripperPose", handTrackingCost, 1e2)
    terminalCostModel.addCost("stateReg", xRegTermCost, 1e-3)
    terminalCostModel.addCost("limitCost", limitCost, 1e3)

    # Create the action model
    dmodelRunning1 = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel2Feet,
                                                                        runningCostModel1)
    dmodelRunning2 = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel1Foot,
                                                                        runningCostModel2)
    dmodelRunning3 = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel1Foot,
                                                                        runningCostModel3)
    dmodelTerminal = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel1Foot,
                                                                        terminalCostModel)

    runningModel1 = crocoddyl.IntegratedActionModelEuler(dmodelRunning1, DT)
    runningModel2 = crocoddyl.IntegratedActionModelEuler(dmodelRunning2, DT)
    runningModel3 = crocoddyl.IntegratedActionModelEuler(dmodelRunning3, DT)
    terminalModel = crocoddyl.IntegratedActionModelEuler(dmodelTerminal, 0)

    # Problem definition
    # x0 = np.concatenate([q0, pin.utils.zero(state.nv)])
    pb = crocoddyl.ShootingProblem(x0, [runningModel1] * T + [runningModel2] * T + [runningModel3] * T, terminalModel)
    
    # Warm-start 
    xs0 = [pb.x0] * (pb.T + 1)
    us0 = pb.quasiStatic([pb.x0] * pb.T)
    
    # Solver
    if(solver_name == 'GNMS'):
        solver = crocoddyl.SolverGNMS(pb)
    else:
        solver = crocoddyl.SolverFDDP(pb)
    
    solver.xs = xs0
    solver.us = us0

    return solver #, xs0, us0

# Problem names
names = ['Pendulum',
         'Kuka',
        #  'Cartpole',  #--> need to explain why it doesn't converge otherwise leave it out 
         'Quadrotor',
         'Humanoid']
# # Initial states 
# if("Pendulum" in names): 
#     pendulum_x0  = np.array([3.14, 0., 0., 0.])
# if("Cartpole" in names): 
#     cartpole_x0  = np.array([0., 3.14, 0., 0.])
# if("Kuka" in names): 
#     kuka_x0      = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.] + [0.]*7)
# if("Quadrotor" in names): 
#     quadrotor    = example_robot_data.load('hector') 
#     quadrotor_x0 = np.array(list(quadrotor.q0) + [0.]*quadrotor.model.nv)  
# if("Humanoid" in names): 
#     humanoid     = example_robot_data.load('talos')
#     humanoid_x0  = np.concatenate([humanoid.model.referenceConfigurations["half_sitting"], pin.utils.zero(humanoid.model.nv)])

N_pb = len(names)


# # Solvers 
# solversGNMS = []
# solversFDDP = []
# if("Pendulum" in names): 
#     solversGNMS.append(create_double_pendulum_problem(pendulum_x0, solver_name='GNMS'))
#     solversFDDP.append(create_double_pendulum_problem(pendulum_x0, solver_name='FDDP'))
# if("Cartpole" in names): 
#     solversGNMS.append(create_cartpole_problem(cartpole_x0, solver_name='GNMS'))
#     solversFDDP.append(create_cartpole_problem(cartpole_x0, solver_name='FDDP'))
# if("Kuka" in names): 
#     solversGNMS.append(create_kuka_problem(kuka_x0, solver_name='GNMS'))
#     solversFDDP.append(create_kuka_problem(kuka_x0, solver_name='FDDP'))
# if("Quadrotor" in names): 
#     solversGNMS.append(create_quadrotor_problem(quadrotor_x0, solver_name='GNMS'))
#     solversFDDP.append(create_quadrotor_problem(quadrotor_x0, solver_name='FDDP'))
# if("Humanoid" in names): 
#     solversGNMS.append(create_humanoid_taichi_problem(humanoid_x0, solver_name='GNMS'))
#     solversFDDP.append(create_humanoid_taichi_problem(humanoid_x0, solver_name='FDDP'))

# print("Created solvers FDDP & GNMS !")


# Compare both solvers with heuristic line-search using KKT condition
MAXITER   = 500 
TOL       = 1e-8 #1e-8
CALLBACKS = False
# KKT_COND  = True

# # Solve fddp
# import matplotlib.pyplot as plt
# fddp_iter = np.zeros((N_pb, 1))
# fddp_kkt = np.zeros((N_pb, 1))
# for k,solver in enumerate(solversFDDP):
#     # Solver setting
#     solver.termination_tolerance = TOL
#     if(CALLBACKS): solver.setCallbacks([crocoddyl.CallbackVerbose()])
#     # solver.use_kkt_criteria = KKT_COND
#     # Warm start & solve
#     print("FDDP solve "+str(k+1)+"/"+str(N_pb)+" : "+names[k])
#     solver.solve(solver.xs, solver.us, MAXITER, False)
#     fddp_iter[k, 0] = solver.iter
#     fddp_kkt[k, 0] = solver.KKT
#     print("iter = ", solver.iter)
#     # if(k==0):
#     #     plt.plot(np.array(solversFDDP[0].xs))
#     #     plt.plot(np.array(solversFDDP[0].us))
#     #     plt.show()

# # Solve GNMS 
# gnms_iter = np.zeros((N_pb, 1))
# gnms_kkt = np.zeros((N_pb, 1))
# for k,solver in enumerate(solversGNMS):
#     solver.termination_tol = TOL
#     solver.with_callbacks = False #CALLBACKS
#     # solver.use_kkt_criteria = KKT_COND
#     solver.use_filter_line_search = True
#     solver.filter_size = MAXITER
#     # Warm start & solve
#     print("GNMS solve "+str(k+1)+"/"+str(N_pb)+" : "+names[k])
#     solver.solve(solver.xs, solver.us, MAXITER, False)
#     gnms_iter[k, 0] = solver.iter
#     gnms_kkt[k, 0] = solver.KKT
#     print("iter = ", solver.iter)

# print("Test results\n")
# for k, name in enumerate(names):
#     print(name+ "_FDDP : "+str(solversFDDP[k].iter))
# for k, name in enumerate(names):
#     print(name+ "_GNMS : "+str(solversGNMS[k].iter))


# Randomize the tests over initial states
SEED = 5
np.random.seed(SEED)
N_samples = 10

# Initial state samples
pendulum_x0_samples  = np.zeros((N_samples, 4))
cartpole_x0_samples  = np.zeros((N_samples, 4))
kuka                 = IiwaConfig.buildRobotWrapper()
kuka_x0_samples      = np.zeros((N_samples, kuka.model.nq + kuka.model.nv))
quadrotor            = example_robot_data.load('hector') 
humanoid             = example_robot_data.load('talos')
quadrotor_x0_samples = np.zeros((N_samples, quadrotor.model.nq + quadrotor.model.nv))
humanoid_x0_samples  = np.zeros((N_samples, humanoid.model.nq + humanoid.model.nv))
for i in range(N_samples):
    pendulum_x0_samples[i,:]  = np.array([np.pi*(2*np.random.rand()-1), 0., 0., 0.])
    cartpole_x0_samples[i,:]  = np.array([0., np.pi*(2*np.random.rand()-1), 0., 0.])
    kuka_x0_samples[i,:]      = np.concatenate([pin.randomConfiguration(kuka.model), np.zeros(kuka.model.nv)])
    quadrotor_x0_samples[i,:] = np.concatenate([pin.randomConfiguration(quadrotor.model), np.zeros(quadrotor.model.nv)])
    humanoid_x0_samples[i,:]  = np.concatenate([pin.randomConfiguration(humanoid.model), np.zeros(humanoid.model.nv)])
print("Created "+str(N_samples)+" random initial states per model !")

# Solvers
solversGNMS = []
solversFDDP = []
for i in range(N_samples):
    # Create problems
    solversGNMS.append([])
    solversFDDP.append([])
    print("---")
    print("Sample "+str(i+1)+'/'+str(N_samples))
    if("Pendulum" in names): 
        solversGNMS[i].append(create_double_pendulum_problem(pendulum_x0_samples[i,:], solver_name='GNMS'))
        solversFDDP[i].append(create_double_pendulum_problem(pendulum_x0_samples[i,:], solver_name='FDDP'))
    if("Cartpole" in names):        
        solversGNMS[i].append(create_cartpole_problem(cartpole_x0_samples[i,:], solver_name='GNMS'))
        solversFDDP[i].append(create_cartpole_problem(cartpole_x0_samples[i,:], solver_name='FDDP'))
    if("Kuka" in names):
        solversGNMS[i].append(create_kuka_problem(kuka_x0_samples[i,:], solver_name='GNMS'))
        solversFDDP[i].append(create_kuka_problem(kuka_x0_samples[i,:], solver_name='FDDP'))
    if("Quadrotor" in names):
        solversGNMS[i].append(create_quadrotor_problem(quadrotor_x0_samples[i,:], solver_name='GNMS'))
        solversFDDP[i].append(create_quadrotor_problem(quadrotor_x0_samples[i,:], solver_name='FDDP'))
    if("Humanoid" in names): 
        solversGNMS[i].append(create_humanoid_taichi_problem(humanoid_x0_samples[i,:], solver_name='GNMS'))
        solversFDDP[i].append(create_humanoid_taichi_problem(humanoid_x0_samples[i,:], solver_name='FDDP'))
print("Created "+str(N_samples)+" solvers per model !")


# Solve fddp
fddp_iter_samples = []  
fddp_kkt_samples  =  []
for i in range(N_samples):
    print("FDDP sample "+str(i+1)+"/"+str(N_samples))
    fddp_iter_samples.append([])
    fddp_kkt_samples.append([])
    for k,solver in enumerate(solversFDDP[i]):
        # Solver setting
        solver.termination_tolerance = TOL
        if(CALLBACKS): solver.setCallbacks([crocoddyl.CallbackVerbose()])
        # solver.use_kkt_criteria = KKT_COND
        # Warm start & solve
        print(" Problem "+str(k+1)+"/"+str(N_pb)+" : "+names[k])
        solver.solve(solver.xs, solver.us, MAXITER, False)
        fddp_iter_samples[i].append(solver.iter)
        fddp_kkt_samples[i].append(solver.KKT)


# Solve GNMS 
gnms_iter_samples = []  
gnms_kkt_samples  =  []
for i in range(N_samples):
    print("GNMS sample "+str(i+1)+"/"+str(N_samples))
    gnms_iter_samples.append([])
    gnms_kkt_samples.append([])
    for k,solver in enumerate(solversGNMS[i]):
        solver.termination_tol = TOL
        solver.with_callbacks = False #CALLBACKS
        # solver.use_kkt_criteria = KKT_COND
        solver.use_filter_line_search = True
        solver.filter_size = MAXITER
        # Warm start & solve
        print(" Problem "+str(k+1)+"/"+str(N_pb)+" : "+names[k])
        solver.solve(solver.xs, solver.us, MAXITER, False)
        gnms_iter_samples[i].append(solver.iter)
        gnms_kkt_samples[i].append(solver.KKT)


# Generate plot of number of iterations for each problem
import matplotlib.pyplot as plt
# Plot 
fig1, ax1 = plt.subplots(1, 1, figsize=(19.2,10.8)) # Err position 
# Average fddp iters
gnms_iter_avg = np.zeros(N_pb)
fddp_iter_avg = np.zeros(N_pb)
gnms_iter_std = np.zeros(N_pb)
fddp_iter_std = np.zeros(N_pb)
for k,exp in enumerate(names):
    fddp_iter_avg[k] = np.mean(np.array(fddp_iter_samples)[:,k])
    gnms_iter_avg[k] = np.mean(np.array(gnms_iter_samples)[:,k]) 
    fddp_iter_std[k] = np.std(np.array(fddp_iter_samples)[:,k])
    gnms_iter_std[k] = np.std(np.array(gnms_iter_samples)[:,k]) 

# Create bar plot
X = np.arange(N_pb)
b1 = ax1.bar(X - 0.13, fddp_iter_avg, yerr=fddp_iter_std, color = 'r', width = 0.25, capsize=10, label='FDDP')
b2 = ax1.bar(X + 0.13, gnms_iter_avg, yerr=gnms_iter_std, color = 'g', width = 0.25, capsize=10, label='GNMS')

# Set axis and stuff
ax1.set_ylabel('Number of iterations', fontsize=26)
ax1.set_ylim(0, 500)
# ax1.set_yticks(X)
ax1.tick_params(axis = 'y', labelsize=22)
# ax1.set_xlabel('Experiment', fontsize=26)
ax1.set_xticks(X)
ax1.set_xticklabels(names, rotation='horizontal', fontsize=18)
ax1.tick_params(axis = 'x', labelsize = 22)

# ax1.set_title('Performance of GNMS and FDDP', fontdict={'size': 26})
ax1.grid(True) 

# Legend 
handles1, labels1 = ax1.get_legend_handles_labels()
fig1.legend(handles1, labels1, loc='upper right', prop={'size': 26}) #, ncols=2)
# Save, show , clean
# fig1.savefig(PREFIX+'pos_err_test_'+str(CUTOFF)+'_new.png')
plt.show()
plt.close('all')




# for i in range(N_pb):
#     if(i == N_pb-1):
#         lab_fddp = 'FDDP' 
#         lab_gnms = 'GNMS'
#     else:
#         lab_fddp = None 
#         lab_gnms = None 
#     ax1.plot(i, fddp_iter_avg[i], marker='o', markerfacecolor='r', linestyle='None', markersize=18, markeredgecolor='k', alpha=1., label=lab_fddp)
#     ax1.plot(i, gnms_iter_avg[i], marker='o', markerfacecolor='b', linestyle='None', markersize=18, markeredgecolor='k', alpha=1., label=lab_gnms)
    
#     for j in range(N_samples):
#         ax1.plot(i, fddp_iter_avg[i], marker='o', markerfacecolor='r', linestyle='None', markersize=12, alpha=0.3)
#         ax1.plot(i, gnms_iter_avg[i], marker='o', markerfacecolor='b', linestyle='None', markersize=12, alpha=0.3)