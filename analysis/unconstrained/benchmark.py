import sys
import numpy as np
import crocoddyl
import pinocchio as pin

from robot_properties_kuka.config import IiwaConfig
import example_robot_data
from cartpole_swingup import DifferentialActionModelCartpole

# Shooting problems helpers

def create_cartpole_problem():
    '''
    Create shooting problem for Cartpole
    '''
    # Creating the DAM for the cartpole
    cartpoleDAM = DifferentialActionModelCartpole()
    cartpoleData = cartpoleDAM.createData()
    cartpoleDAM = model = DifferentialActionModelCartpole()
    # Using NumDiff for computing the derivatives. We specify the
    # withGaussApprox=True to have approximation of the Hessian based on the
    # Jacobian of the cost residuals.
    cartpoleND = crocoddyl.DifferentialActionModelNumDiff(cartpoleDAM, True)
    # Getting the IAM using the simpletic Euler rule
    timeStep = 5e-2
    cartpoleIAM = crocoddyl.IntegratedActionModelEuler(cartpoleND, timeStep)
    # Creating the shooting problem
    x0 = np.array([0., 3.14, 0., 0.])
    T = 50
    terminalCartpole = DifferentialActionModelCartpole()
    terminalCartpoleDAM = crocoddyl.DifferentialActionModelNumDiff(terminalCartpole, True)
    terminalCartpoleIAM = crocoddyl.IntegratedActionModelEuler(terminalCartpoleDAM)
    terminalCartpole.costWeights[0] = 200
    terminalCartpole.costWeights[1] = 200
    terminalCartpole.costWeights[2] = 1.
    terminalCartpole.costWeights[3] = 0.1
    terminalCartpole.costWeights[4] = 0.01
    terminalCartpole.costWeights[5] = 0.0001
    return crocoddyl.ShootingProblem(x0, [cartpoleIAM] * T, terminalCartpoleIAM)


def create_kuka_problem():
    '''
    Create shooting problem for kuka reaching task
    '''
    robot = IiwaConfig.buildRobotWrapper()
    model = robot.model
    nq = model.nq; nv = model.nv; nu = nq; nx = nq+nv
    q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.])
    v0 = np.zeros(nv)
    x0 = np.concatenate([q0, v0]).copy()
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
    T = 10
    return crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)


def create_quadrotor_problem():
    '''
    Create shooting problem for quadrotor task
    '''
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
    x0 = np.concatenate([hector.q0, np.zeros(state.nv)])
    return crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)


def create_humanoid_taichi_problem():
    '''
    Create shooting problem for Talos taichi task
    '''
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
    x0 = np.concatenate([q0, pin.utils.zero(state.nv)])
    return crocoddyl.ShootingProblem(x0, [runningModel1] * T + [runningModel2] * T + [runningModel3] * T, terminalModel)


# Create problems 
problems = \
        [
        create_cartpole_problem(),
        create_kuka_problem(),
        create_quadrotor_problem(),
        create_humanoid_taichi_problem()
        ]

# Warm-starts = quasi-static
xss = []
uss = []
for pb in problems:
    xss.append([pb.x0] * (pb.T + 1))
    uss.append(pb.quasiStatic([pb.x0] * pb.T))
# faire while (tant que par pire que quelqu'un) avec max 10 antécédents (ou maxit => tous)
# Solvers 
solversGNMS = []
solversFDDP = []
for pb in problems:
    solversGNMS.append(crocoddyl.SolverGNMS(pb))
    solversFDDP.append(crocoddyl.SolverFDDP(pb))


# Compare both solvers with heuristic line-search using KKT condition
MAXITER   = 500
TOL       = 1e-6
CALLBACKS = False
KKT_COND  = True

for solver in solversFDDP:
    solver.termination_tolerance = TOL
    solver.with_callbacks = CALLBACKS
    solver.use_kkt_criteria = KKT_COND
    solver.use
print("Solver GNMS :")  
# print("  Use KKT condition     = ", solverGNMS.use_kkt_condition)
print("  Termination tolerance = ", solverGNMS.termination_tolerance)
print("  Penalty parameter mu  = ", solverGNMS.mu)
print("  use_heuristic_line_search  = ", solverGNMS.use_heuristic_line_search)


# FDDP 
solverFDDP = crocoddyl.SolverFDDP(problem)
solverFDDP.termination_tolerance = TOL
if(CALLBACKS): solverFDDP.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
solverFDDP.use_kkt_condition = KKT_COND
print("Solver FDPP :")  
print("  Use KKT condition     = ", solverFDDP.use_kkt_condition)
print("  Termination tolerance = ", solverFDDP.termination_tolerance)


# SOLVE GNMS
mu_values      = [1e0, 1e2, 1e3]
converged_gnms = []
iter_gnms      = []
for mu in mu_values:
    solverGNMS.mu = mu
    print('------------------')
    print("mu = ", solverGNMS.mu)
    converged_gnms.append(solverGNMS.solve(xs, us, MAXITER, False))
    print("nb iter = ", solverGNMS.iter)
    iter_gnms.append(solverGNMS.iter)
    print('------------------')
    
# SOLVE FDDP
converged_fddp = solverFDDP.solve(xs, us, MAXITER, False)
iter_fddp      = solverFDDP.iter

# Print
print("GNMS mu \n", mu_values)
print("GNMS iter \n", iter_gnms)
print("GNMS converged \n", converged_gnms)
print("FDDP iter \n", iter_fddp)
print("FDDP converged \n", converged_fddp)
