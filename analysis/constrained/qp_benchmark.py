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

>>> becnhmarks for rebutal :
    - osqp dense vs stagewise (ours)
    - reply "look at S. Caron osqp (dense) > hpipm (dense) to reviewer
    - remove any mention to hpipm in the paper
Randomizing over initial states
'''
import numpy as np
import crocoddyl
import pinocchio as pin

from mim_robots.robot_loader import load_pinocchio_wrapper
import example_robot_data
# from unconstrained.bench_utils.cartpole_swingup import DifferentialActionModelCartpole
from crocoddyl.utils.pendulum import CostModelDoublePendulum, ActuationModelDoublePendulum
import mim_solvers

import pathlib
import os
python_path = pathlib.Path('/home/ajordana/workspace/mim_solvers/python/').absolute()
os.sys.path.insert(1, str(python_path))
from csqp import CSQP

import time

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
    robot = load_pinocchio_wrapper("iiwa", locked_joints=["A7"])
    model = robot.model
    # State and actuation model
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFull(state)

    # Create cost terms
    # Control regularization cost
    uResidual = crocoddyl.ResidualModelContactControlGrav(state)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    # State regularization cost
    xResidual = crocoddyl.ResidualModelState(state, x0)
    xRegCost = crocoddyl.CostModelResidual(state, xResidual)
    # endeff frame translation cost
    endeff_frame_id = model.getFrameId("A3")
    endeff_translation = np.array([0.5, 0.1, 0.2]) 
    frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
        state, endeff_frame_id, endeff_translation
    )
    frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)
    # ee velocity cost
    frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(
        state, endeff_frame_id, pin.Motion.Zero(), pin.LOCAL_WORLD_ALIGNED
    )
    frameVelocityCost = crocoddyl.CostModelResidual(state, frameVelocityResidual)
    # Create contraint on end-effector (small box around initial EE position)
    frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
        state, endeff_frame_id, np.zeros(3)
    )
    data = model.createData()
    pin.framesForwardKinematics(model, data, x0[:model.nq])
    p0 = data.oMf[endeff_frame_id].translation
    ee_contraint = crocoddyl.ConstraintModelResidual(
        state,
        frameTranslationResidual,
        p0 - np.array([10.5, 0.5, 10.5]),
        p0 + np.array([0.5, 10.5, 10.5]),
    )
    #  Constraint on frame velocity
    frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(
        state, endeff_frame_id, pin.Motion.Zero(), pin.LOCAL_WORLD_ALIGNED
    )
    # ee_vel_constraint = crocoddyl.ConstraintModelResidual(
    #     state,
    #     frameVelocityResidual,
    #     -np.array([10.]*6),
    #     np.array([10.]*6),
    # )
    xlb = np.concatenate([robot.model.lowerPositionLimit, -robot.model.velocityLimit])
    xub = np.concatenate([robot.model.upperPositionLimit, robot.model.velocityLimit])
    xLimitResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
    constraintState = crocoddyl.ConstraintModelResidual(state, xLimitResidual, xlb, xub)

    # Contact model 
    contactModel = crocoddyl.ContactModelMultiple(state, actuation.nu)
    contact_frame_id1 = model.getFrameId("contact") 
    contact_frame_id2 = model.getFrameId("A5") 
    contactModel.addContact("contact1", crocoddyl.ContactModel6D(state, contact_frame_id1, pin.SE3.Identity(), pin.LOCAL, np.array([0., 50.])) , active=True)
    contactModel.addContact("contact2", crocoddyl.ContactModel6D(state, contact_frame_id2, pin.SE3.Identity(), pin.LOCAL, np.array([0., 50.])) , active=True)
    # End-effector frame force cost
    frameForceResidual1 = crocoddyl.ResidualModelContactForce(state, contact_frame_id1, pin.Force.Zero(), 6, actuation.nu)
    frameForceResidual2 = crocoddyl.ResidualModelContactForce(state, contact_frame_id2, pin.Force.Zero(), 6, actuation.nu)
    contactForceCost1 = crocoddyl.CostModelResidual(state, frameForceResidual1)
    contactForceCost2 = crocoddyl.CostModelResidual(state, frameForceResidual2)
    constraintForce1 = crocoddyl.ConstraintModelResidual(state, frameForceResidual1, np.array([0., 0, 0]*2), np.array([np.inf, np.inf, np.inf]*2))
    constraintForce2 = crocoddyl.ConstraintModelResidual(state, frameForceResidual2, np.array([0., 0, 0]*2), np.array([np.inf, np.inf, np.inf]*2))
    # Create the running models
    runningModels = []
    dt = 1e-2
    T = 50
    for t in range(T + 1):
        runningCostModel = crocoddyl.CostModelSum(state)
        # Add costs
        runningCostModel.addCost("stateReg", xRegCost, 1e-1)
        runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
        if t != T:
            runningCostModel.addCost("translation", frameTranslationCost, 1)
            # runningCostModel.addCost("force1", contactForceCost1, 1e-3)
            # runningCostModel.addCost("force2", contactForceCost2, 1e-3)
            runningCostModel.addCost("velocity", frameVelocityCost, 1e-3)
        else:
            runningCostModel.addCost("translation", frameTranslationCost, 1)
            runningCostModel.addCost("velocity", frameVelocityCost, 1e-1)
        # Define contraints
        constraints = crocoddyl.ConstraintModelManager(state, actuation.nu)
        if t != 0:
            # constraints.addConstraint("ee_bound", ee_contraint)
            # constraints.addConstraint("xLim", constraintState)
            constraints.addConstraint("force1", constraintForce1)
            # constraints.addConstraint("force2", constraintForce2)
        # Create Differential action model
        running_DAM = crocoddyl.DifferentialActionModelContactFwdDynamics(
            state, actuation, contactModel, runningCostModel, constraints, inv_damping=0., enable_force=True)
        # Apply Euler integration
        running_model = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
        runningModels.append(running_model)        
    pb = crocoddyl.ShootingProblem(x0, runningModels[:-1], runningModels[-1])
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

def create_humanoid_taichi_problem(target=np.array([0.4, 0, 1.2]), 
                                   JOINT_CONSTRAINT=False, 
                                   FORCE_COST=True,
                                   FORCE_CONSTRAINT=True):
    '''
    Create shooting problem for Talos taichi task
    '''
    # Load robot
    robot = example_robot_data.load("talos")
    rmodel = robot.model
    lims = rmodel.effortLimit
    # lims[19:] *= 0.5  # reduced artificially the torque limits
    rmodel.effortLimit = lims
    # Create data structures
    rdata = rmodel.createData()
    state = crocoddyl.StateMultibody(rmodel)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    # Set integration time
    DT = 5e-2
    T = 10
    # Initialize reference state, target and reference CoM
    rightFoot = "right_sole_link"
    leftFoot = "left_sole_link"
    endEffector = "gripper_left_joint"
    endEffectorId = rmodel.getFrameId(endEffector)
    rightFootId = rmodel.getFrameId(rightFoot)
    leftFootId = rmodel.getFrameId(leftFoot)
    q0 = rmodel.referenceConfigurations["half_sitting"]
    x0 = np.concatenate([q0, np.zeros(rmodel.nv)])
    pin.forwardKinematics(rmodel, rdata, q0)
    pin.updateFramePlacements(rmodel, rdata)
    rfPos0 = rdata.oMf[rightFootId].translation
    lfPos0 = rdata.oMf[leftFootId].translation
    comRef = (rfPos0 + lfPos0) / 2
    comRef[2] = pin.centerOfMass(rmodel, rdata, q0)[2].item()
    # Create two contact models used along the motion
    contactModel1Foot = crocoddyl.ContactModelMultiple(state, actuation.nu)
    contactModel2Feet = crocoddyl.ContactModelMultiple(state, actuation.nu)
    supportContactModelLeft = crocoddyl.ContactModel6D(
        state,
        leftFootId,
        pin.SE3.Identity(),
        pin.LOCAL,
        actuation.nu,
        np.array([0, 40]),
    )
    supportContactModelRight = crocoddyl.ContactModel6D(
        state,
        rightFootId,
        pin.SE3.Identity(),
        pin.LOCAL,
        actuation.nu,
        np.array([0, 40]),
    )
    contactModel1Foot.addContact(rightFoot + "_contact", supportContactModelRight)
    contactModel2Feet.addContact(leftFoot + "_contact", supportContactModelLeft)
    contactModel2Feet.addContact(rightFoot + "_contact", supportContactModelRight)
    # Cost for self-collision
    maxfloat = np.inf #sys.float_info.max !!! needs to be inf otherwise HPIPM would crash
    xlb = np.concatenate(
        [
            -maxfloat * np.ones(6),  # dimension of the SE(3) manifold
            rmodel.lowerPositionLimit[7:],
            -maxfloat * np.ones(state.nv),
        ]
    )
    xub = np.concatenate(
        [
            maxfloat * np.ones(6),  # dimension of the SE(3) manifold
            rmodel.upperPositionLimit[7:],
            maxfloat * np.ones(state.nv),
        ]
    )
    bounds = crocoddyl.ActivationBounds(xlb, xub, 1.0)
    xLimitResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
    xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
    limitCost = crocoddyl.CostModelResidual(state, xLimitActivation, xLimitResidual)
    # Cost for state and control
    xResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
    xActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [10] * state.nv) ** 2
    )
    uResidual = crocoddyl.ResidualModelControl(state, actuation.nu)
    xTActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [100] * state.nv) ** 2
    )
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    xRegTermCost = crocoddyl.CostModelResidual(state, xTActivation, xResidual)

    # Cost for target reaching: hand and foot
    handTrackingResidual = crocoddyl.ResidualModelFramePlacement(
        state, endEffectorId, pin.SE3(np.eye(3), target), actuation.nu
    )
    handTrackingActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([1] * 3 + [0.0001] * 3) ** 2
    )
    handTrackingCost = crocoddyl.CostModelResidual(
        state, handTrackingActivation, handTrackingResidual
    )

    footTrackingResidual = crocoddyl.ResidualModelFramePlacement(
        state, leftFootId, pin.SE3(np.eye(3), np.array([0.0, 0.4, 0.0])), actuation.nu
    )
    footTrackingActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([1, 1, 0.1] + [1.0] * 3) ** 2
    )
    footTrackingCost1 = crocoddyl.CostModelResidual(
        state, footTrackingActivation, footTrackingResidual
    )
    footTrackingResidual = crocoddyl.ResidualModelFramePlacement(
        state,
        leftFootId,
        pin.SE3(np.eye(3), np.array([0.3, 0.15, 0.35])),
        actuation.nu,
    )
    footTrackingCost2 = crocoddyl.CostModelResidual(
        state, footTrackingActivation, footTrackingResidual
    )
    # Cost for CoM reference
    # comResidual = crocoddyl.ResidualModelCoMPosition(state, comRef, actuation.nu)
    # comTrack = crocoddyl.CostModelResidual(state, comResidual)
    # Create cost model per each action model. We divide the motion in 3 phases plus its
    # terminal model.
    runningCostModel1 = crocoddyl.CostModelSum(state, actuation.nu)
    runningCostModel2 = crocoddyl.CostModelSum(state, actuation.nu)
    runningCostModel3 = crocoddyl.CostModelSum(state, actuation.nu)
    terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)
    # Then let's added the running and terminal cost functions
    runningCostModel1.addCost("gripperPose", handTrackingCost, 1e2)
    runningCostModel1.addCost("stateReg", xRegCost, 1e-3)
    runningCostModel1.addCost("ctrlReg", uRegCost, 1e-4)
    if not JOINT_CONSTRAINT:
        runningCostModel1.addCost("limitCost", limitCost, 1e3)
    runningCostModel2.addCost("gripperPose", handTrackingCost, 1e2)
    runningCostModel2.addCost("footPose", footTrackingCost1, 1e1)
    runningCostModel2.addCost("stateReg", xRegCost, 1e-3)
    runningCostModel2.addCost("ctrlReg", uRegCost, 1e-4)
    if not JOINT_CONSTRAINT:
        runningCostModel2.addCost("limitCost", limitCost, 1e3)
    runningCostModel3.addCost("gripperPose", handTrackingCost, 1e2)
    runningCostModel3.addCost("footPose", footTrackingCost2, 1e1)
    runningCostModel3.addCost("stateReg", xRegCost, 1e-3)
    runningCostModel3.addCost("ctrlReg", uRegCost, 1e-4)
    if not JOINT_CONSTRAINT:
        runningCostModel3.addCost("limitCost", limitCost, 1e3)
    terminalCostModel.addCost("gripperPose", handTrackingCost, 1e2)
    terminalCostModel.addCost("stateReg", xRegTermCost, 1e-3)
    if not JOINT_CONSTRAINT:
        terminalCostModel.addCost("limitCost", limitCost, 1e3)
    constraintModelManager = crocoddyl.ConstraintModelManager(state, actuation.nu)
    fref = pin.Force.Zero()
    ForceResidual = crocoddyl.ResidualModelContactForce(state, rightFootId, fref, 6, actuation.nu)
    if FORCE_COST:
        Forcecost = crocoddyl.CostModelResidual(state, ForceResidual)
        runningCostModel1.addCost("forcecost1", Forcecost, 1e-3)
        runningCostModel2.addCost("forcecost2", Forcecost, 1e-3)
        runningCostModel3.addCost("forcecost3", Forcecost, 1e-3)
    if FORCE_CONSTRAINT:
        constraintForce = crocoddyl.ConstraintModelResidual(state, ForceResidual, np.array([0., 0, 0]*2), np.array([np.inf, np.inf, np.inf]*2))
        constraintModelManager.addConstraint("force", constraintForce)
    if JOINT_CONSTRAINT:
        constraintState = crocoddyl.ConstraintModelResidual(state, xLimitResidual, xlb, xub)
        constraintModelManager.addConstraint("state", constraintState)
    # Create the action model
    dmodelRunning1 = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contactModel2Feet, runningCostModel1, constraintModelManager, 0., True)
    dmodelRunning2 = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contactModel1Foot, runningCostModel2, constraintModelManager, 0., True)
    dmodelRunning3 = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contactModel1Foot, runningCostModel3, constraintModelManager, 0., True)
    dmodelTerminal = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contactModel1Foot, terminalCostModel
    )
    runningModel1 = crocoddyl.IntegratedActionModelEuler(dmodelRunning1, DT)
    runningModel2 = crocoddyl.IntegratedActionModelEuler(dmodelRunning2, DT)
    runningModel3 = crocoddyl.IntegratedActionModelEuler(dmodelRunning3, DT)
    terminalModel = crocoddyl.IntegratedActionModelEuler(dmodelTerminal, 0)
    # Problem definition
    pb = crocoddyl.ShootingProblem(
        x0, [runningModel1] * T + [runningModel2] * T + [runningModel3] * T, terminalModel
    )
    return pb


# Solver params
MAXITER     = 1     
TOL         = 1e-4
CALLBACKS   = False
MAX_QP_ITER = 50000
MAX_QP_TIME = int(100*1e3) # in ms
EPS_ABS     = 1e-1
EPS_REL     = 0.
SAVE        = False # Save figure 

# Benchmark params
SEED = 10 ; np.random.seed(SEED)
N_samples = 100
names = [
    #    'Pendulum'] # maxiter = 500
        #  'Kuka'] # maxiter = 100
         'Taichi'] #
        # #  'Cartpole']  #--> need to explain why it doesn't converge otherwise leave it out 
        #  'Quadrotor'] # maxiter = 200

N_pb = len(names)

# Solvers
SOLVERS = ['CSQP',
        #    'OSQP',
        #    'HPIPM_DENSE', 
           'HPIPM_OCP']

if('CSQP' in SOLVERS):
    solversCSQP         = [] 
    csqp_iter_samples   = []  
    csqp_time_samples   = []
    csqp_solved_samples = []
if('OSQP' in SOLVERS):
    solversOSQP         = [] 
    osqp_iter_samples   = []  
    osqp_time_samples   = []
    osqp_solved_samples = []
if('HPIPM_DENSE' in SOLVERS):
    solversHPIPM_dense         = [] 
    hpipm_dense_iter_samples   = []  
    hpipm_dense_time_samples   = []
    hpipm_dense_solved_samples = []
if('HPIPM_OCP' in SOLVERS):
    solversHPIPM_ocp         = [] 
    hpipm_ocp_iter_samples   = []  
    hpipm_ocp_time_samples   = []
    hpipm_ocp_solved_samples = []

# Initial states
pendulum_x0  = np.array([3.14, 0., 0., 0.])
cartpole_x0  = np.array([0., 3.14, 0., 0.])
kuka_x0      = np.array([0.1, 0.2, 0., 0., -0.2, 0.2] + [0.]*6)
quadrotor    = example_robot_data.load('hector') 
quadrotor_x0 = np.array(list(quadrotor.q0) + [0.]*quadrotor.model.nv) 
taichi_p0    = np.array([0.4, 0, 1.2])

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
    if(name == "Taichi"): 
        pb = create_humanoid_taichi_problem(taichi_p0) 

    # Create solver CSQP 
    if('CSQP' in SOLVERS):
        solvercsqp = mim_solvers.SolverCSQP(pb)
        # solvercsqp.xs = [solvercsqp.problem.x0] * (solvercsqp.problem.T + 1)  
        # solvercsqp.us = [solvercsqp.problem.quasiStatic([solvercsqp.problem.x0] * solvercsqp.problem.T)
        solvercsqp.termination_tolerance = TOL
        solvercsqp.max_qp_iters = MAX_QP_ITER
        solvercsqp.with_qp_callbacks = False
        solvercsqp.eps_abs = EPS_ABS
        solvercsqp.eps_rel = EPS_REL
        # print("CHECK = ", solvercsqp.equality_qp_initial_guess)
        solvercsqp.equality_qp_initial_guess = False
        # assert(solvercsqp.equality_qp_initial_guess == False)
        # print("ASSERT OK")
        solvercsqp.update_rho_with_heuristic = False
        solvercsqp.with_callbacks = CALLBACKS
        # solvercsqp.with_qp_callbacks = True # CALLBACKS
        solversCSQP.append(solvercsqp)
    
    # Create solver OSQP
    if('OSQP' in SOLVERS):
        solverosqp = CSQP(pb, "OSQP")
        solverosqp.xs = [solverosqp.problem.x0] * (solverosqp.problem.T + 1)  
        solverosqp.us = solverosqp.problem.quasiStatic([solverosqp.problem.x0] * solverosqp.problem.T)
        solverosqp.termination_tolerance = TOL
        solverosqp.max_qp_iters = MAX_QP_ITER
        solverosqp.eps_abs = EPS_ABS
        solverosqp.eps_rel = EPS_REL
        solverosqp.with_callbacks = CALLBACKS
        solversOSQP.append(solverosqp)

    # Create solver HPIPM dense
    if('HPIPM_DENSE' in SOLVERS):
        solverhpipm_dense = CSQP(pb, "HPIPM_DENSE")
        solverhpipm_dense.xs = [solverhpipm_dense.problem.x0] * (solverhpipm_dense.problem.T + 1)  
        solverhpipm_dense.us = solverhpipm_dense.problem.quasiStatic([solverhpipm_dense.problem.x0] * solverhpipm_dense.problem.T)
        solverhpipm_dense.termination_tolerance  = TOL
        solverhpipm_dense.max_qp_iters = 1 #MAX_QP_ITER
        solverhpipm_dense.eps_abs = EPS_ABS
        solverhpipm_dense.eps_rel = EPS_REL
        solverhpipm_dense.with_callbacks = CALLBACKS
        solversHPIPM_dense.append(solverhpipm_dense)

    # Create solver HPIPM ocp
    if('HPIPM_OCP' in SOLVERS):
        solverhpipm_ocp = CSQP(pb, "HPIPM_OCP")
        # solverhpipm_ocp.xs = [solverhpipm_ocp.problem.x0] * (solverhpipm_ocp.problem.T + 1)  
        # solverhpipm_ocp.us = solverhpipm_ocp.problem.quasiStatic([solverhpipm_ocp.problem.x0] * solverhpipm_ocp.problem.T)
        solverhpipm_ocp.termination_tolerance  = TOL
        solverhpipm_ocp.max_qp_iters = MAX_QP_ITER
        solverhpipm_ocp.eps_abs = EPS_ABS
        solverhpipm_ocp.eps_rel = EPS_REL
        solverhpipm_ocp.with_callbacks = CALLBACKS
        solversHPIPM_ocp.append(solverhpipm_ocp)


# Initial state samples
pendulum_x0_samples  = np.zeros((N_samples, 4))
cartpole_x0_samples  = np.zeros((N_samples, 4))
kuka                 = load_pinocchio_wrapper("iiwa", locked_joints=["A7"])
kuka_x0_samples      = np.zeros((N_samples, kuka.model.nq + kuka.model.nv))
quadrotor            = example_robot_data.load('hector') 
quadrotor_x0_samples = np.zeros((N_samples, quadrotor.model.nq + quadrotor.model.nv))
taichi_p0_samples  = np.zeros((N_samples, 3))
for i in range(N_samples):
    pendulum_x0_samples[i,:]  = np.array([np.pi*(2*np.random.rand()-1), 0., 0., 0.])
    cartpole_x0_samples[i,:]  = np.array([0., np.pi/2, 0., 0.])
    kuka_x0_samples[i,:]      = np.concatenate([pin.randomConfiguration(kuka.model), np.zeros(kuka.model.nv)])
    quadrotor_x0_samples[i,:] = np.concatenate([pin.randomConfiguration(quadrotor.model), np.zeros(quadrotor.model.nv)])
    quadrotor_x0_samples[i,:] = np.concatenate([pin.randomConfiguration(quadrotor.model), np.zeros(quadrotor.model.nv)])
    err = np.zeros(3)
    err[2] = 2*np.random.rand(1) - 1
    taichi_p0_samples[i,:]  = taichi_p0 + 0.05*err

print("Created "+str(N_samples)+" random initial states per model !")

# Solve problems for sample initial states
for i in range(N_samples):
    if('CSQP' in SOLVERS):
        csqp_iter_samples.append([])
        csqp_time_samples.append([])
        csqp_solved_samples.append([])

    if('OSQP' in SOLVERS):
        osqp_iter_samples.append([])
        osqp_time_samples.append([])
        osqp_solved_samples.append([])

    if('HPIPM_DENSE' in SOLVERS):
        hpipm_dense_iter_samples.append([])
        hpipm_dense_time_samples.append([])
        hpipm_dense_solved_samples.append([])

    if('HPIPM_OCP' in SOLVERS):
        hpipm_ocp_iter_samples.append([])
        hpipm_ocp_time_samples.append([])
        hpipm_ocp_solved_samples.append([])

    print("---")
    print("Sample "+str(i+1)+'/'+str(N_samples))
    for k,name in enumerate(names):
        # Initial state
        if(name == "Pendulum"):  x0 = pendulum_x0_samples[i,:]
        if(name == "Cartpole"):  x0 = cartpole_x0_samples[i,:]
        if(name == "Kuka"):      x0 = kuka_x0_samples[i,:]
        if(name == "Quadrotor"): x0 = quadrotor_x0_samples[i,:]
        if(name == "Taichi"):    p0 = taichi_p0_samples[i,:]

        # CSQP
        if('CSQP' in SOLVERS):
            print("   Problem : "+name+" CSQP")
            solvercsqp = solversCSQP[k]
            if(name == "Taichi"):
                models = list(solvercsqp.problem.runningModels) + [solvercsqp.problem.terminalModel]
                for m in models: m.differential.costs.costs["gripperPose"].cost.residual.reference = pin.SE3(np.eye(3), p0.copy())
            else:
                solvercsqp.problem.x0 = x0
            solvercsqp.xs = [solvercsqp.problem.x0] * (solvercsqp.problem.T + 1) 
            solvercsqp.us = solvercsqp.problem.quasiStatic([solvercsqp.problem.x0] * solvercsqp.problem.T)
            solvercsqp.solve(solvercsqp.xs, solvercsqp.us, 0, False)
            t1 = time.time()
            solvercsqp.computeDirection(True)
            solvercsqp.qp_time = time.time() - t1
            # Check convergence and fill data 
            solved = (solvercsqp.norm_primal < EPS_ABS and solvercsqp.norm_dual < EPS_ABS and solvercsqp.qp_iters <= MAX_QP_ITER)
            csqp_solved_samples[i].append( solved )
            if(not solved): 
                print("      FAILED !!!!")
                csqp_iter_samples[i].append(MAX_QP_ITER)
                csqp_time_samples[i].append(MAX_QP_TIME)
            else:
                csqp_iter_samples[i].append(solvercsqp.qp_iters)
                csqp_time_samples[i].append(solvercsqp.qp_time*1e3)
                print("     QP Time = ", solvercsqp.qp_time)
                print("     QP Iter = ", solvercsqp.qp_iters)
            # print(" - Primal residual: ", solvercsqp.norm_primal)
            # print(" - Dual residual: ", solvercsqp.norm_dual)
        
        # OSQP
        if('OSQP' in SOLVERS):
            print("   Problem : "+name+" OSQP")
            solverosqp = solversOSQP[k]
            if(name == "Taichi"):
                models = list(solvercsqp.problem.runningModels) + [solvercsqp.problem.terminalModel]
                for m in models: m.differential.costs.costs["gripperPose"].cost.residual.reference = pin.SE3(np.eye(3), p0.copy())
            else:
                solvercsqp.problem.x0 = x0
            solverosqp.xs = [solverosqp.problem.x0] * (solverosqp.problem.T + 1) 
            solverosqp.us = solverosqp.problem.quasiStatic([solverosqp.problem.x0] * solverosqp.problem.T)
            solverosqp.solve(solverosqp.xs, solverosqp.us, MAXITER, False)
            solved = (solverosqp.found_qp_sol and solverosqp.norm_primal < EPS_ABS and solverosqp.norm_dual < EPS_ABS and solverosqp.qp_iters <= MAX_QP_ITER)
            osqp_solved_samples[i].append( solved )
            if(not solved): 
                print("      FAILED !!!!")
                osqp_iter_samples[i].append(MAX_QP_ITER)
                osqp_time_samples[i].append(MAX_QP_TIME)
            else:
                osqp_iter_samples[i].append(solverosqp.qp_iters)
                osqp_time_samples[i].append(solverosqp.qp_time*1e3)
                print("     QP Time = ", solverosqp.qp_time)
                print("     QP Iter = ", solverosqp.qp_iters)

        # HPIPM_DENSE
        if('HPIPM_DENSE' in SOLVERS):
            print("   Problem : "+name+" HPIPM_DENSE")
            solverhpipm_dense = solversHPIPM_dense[k]
            if(name == "Taichi"):
                models = list(solvercsqp.problem.runningModels) + [solvercsqp.problem.terminalModel]
                for m in models: m.differential.costs.costs["gripperPose"].cost.residual.reference = pin.SE3(np.eye(3), p0.copy())
            else:
                solvercsqp.problem.x0 = x0
            solverhpipm_dense.xs = [solverhpipm_dense.problem.x0] * (solverhpipm_dense.problem.T + 1) 
            solverhpipm_dense.us = solverhpipm_dense.problem.quasiStatic([solverhpipm_dense.problem.x0] * solverhpipm_dense.problem.T)
            # solverhpipm_dense.solve(solverhpipm_dense.xs, solverhpipm_dense.us, MAXITER, False)
            solverhpipm_dense.found_qp_sol = False
            if(solverhpipm_dense.found_qp_sol):
                solved = (solverhpipm_dense.norm_primal < EPS_ABS and solverhpipm_dense.norm_dual < EPS_ABS and solverhpipm_dense.qp_iters <= MAX_QP_ITER)
            else:
                solved = False
            solved = False
            hpipm_dense_solved_samples[i].append( solved )
            if(not solved): 
                print("      FAILED !!!!")
                hpipm_dense_iter_samples[i].append(MAX_QP_ITER)
                hpipm_dense_time_samples[i].append(MAX_QP_TIME)
            else:
                hpipm_dense_iter_samples[i].append(solverhpipm_dense.qp_iters)
                hpipm_dense_time_samples[i].append(solverhpipm_dense.qp_time*1e3)
                print("     QP Time = ", solverhpipm_dense.qp_time)
                print("     QP Iter = ", solverhpipm_dense.qp_iters)

        # HPIPM_OCP    
        if('HPIPM_OCP' in SOLVERS):    
            print("   Problem : "+name+" HPIPM_OCP")
            solverhpipm_ocp = solversHPIPM_ocp[k]
            if(name == "Taichi"):
                models = list(solvercsqp.problem.runningModels) + [solvercsqp.problem.terminalModel]
                for m in models: m.differential.costs.costs["gripperPose"].cost.residual.reference = pin.SE3(np.eye(3), p0.copy())
            else:
                solvercsqp.problem.x0 = x0
            solverhpipm_ocp.xs = [solverhpipm_ocp.problem.x0] * (solverhpipm_ocp.problem.T + 1) 
            solverhpipm_ocp.us = solverhpipm_ocp.problem.quasiStatic([solverhpipm_ocp.problem.x0] * solverhpipm_ocp.problem.T)
            # import pdb ; pdb.set_trace()
            solverhpipm_ocp.solve(solverhpipm_ocp.xs, solverhpipm_ocp.us, MAXITER, False)
                # Check convergence
            if(solverhpipm_ocp.found_qp_sol):
                solved = (solverhpipm_ocp.norm_primal < EPS_ABS and solverhpipm_ocp.norm_dual < EPS_ABS and solverhpipm_ocp.qp_iters <= MAX_QP_ITER)
            else:
                solved = False
            hpipm_ocp_solved_samples[i].append( solved )
            if(not solved): 
                print("      FAILED !!!!")
                hpipm_ocp_iter_samples[i].append(MAX_QP_ITER)
                hpipm_ocp_time_samples[i].append(MAX_QP_TIME)
            else:
                hpipm_ocp_iter_samples[i].append(solverhpipm_ocp.qp_iters)
                hpipm_ocp_time_samples[i].append(solverhpipm_ocp.qp_time*1e3)
            print("     QP Time = ", solverhpipm_ocp.qp_time)
            print("     QP Iter = ", solverhpipm_ocp.qp_iters)

# Compute convergence statistics
if('CSQP' in SOLVERS):  
    csqp_iter_solved = np.zeros((MAX_QP_ITER, N_pb))
    csqp_time_solved = np.zeros((MAX_QP_TIME, N_pb))
if('OSQP' in SOLVERS):  
    osqp_iter_solved = np.zeros((MAX_QP_ITER, N_pb))
    osqp_time_solved = np.zeros((MAX_QP_TIME, N_pb))
if('HPIPM_DENSE' in SOLVERS):  
    hpipm_dense_iter_solved = np.zeros((MAX_QP_ITER, N_pb))
    hpipm_dense_time_solved = np.zeros((MAX_QP_TIME, N_pb))
if('HPIPM_OCP' in SOLVERS):  
    hpipm_ocp_iter_solved = np.zeros((MAX_QP_ITER, N_pb))
    hpipm_ocp_time_solved = np.zeros((MAX_QP_TIME, N_pb))
for k,exp in enumerate(names):
    # Count number of problems solved for each sample initial state 
    for i in range(N_samples):
        # For sample i of problem k , compare nb iter to max iter
        if('CSQP' in SOLVERS): 
            csqp_iter_ik  = np.array(csqp_iter_samples)[i,k]
            csqp_time_ik  = np.array(csqp_time_samples)[i,k]
        if('OSQP' in SOLVERS): 
            osqp_iter_ik = np.array(osqp_iter_samples)[i,k]
            osqp_time_ik = np.array(osqp_time_samples)[i,k]
        if('HPIPM_DENSE' in SOLVERS): 
            hpipm_dense_iter_ik = np.array(hpipm_dense_iter_samples)[i,k]
            hpipm_dense_time_ik = np.array(hpipm_dense_time_samples)[i,k]
        if('HPIPM_OCP' in SOLVERS): 
            hpipm_ocp_iter_ik = np.array(hpipm_ocp_iter_samples)[i,k]
            hpipm_ocp_time_ik = np.array(hpipm_ocp_time_samples)[i,k]
        # Number of iterations
        for j in range(MAX_QP_ITER):
            if('CSQP' in SOLVERS): 
                if(csqp_iter_ik < j): csqp_iter_solved[j,k] += 1
            if('OSQP' in SOLVERS): 
                if(osqp_iter_ik < j): osqp_iter_solved[j,k] += 1
            if('HPIPM_DENSE' in SOLVERS): 
                if(hpipm_dense_iter_ik < j): hpipm_dense_iter_solved[j,k] += 1
            if('HPIPM_OCP' in SOLVERS): 
                if(hpipm_ocp_iter_ik < j): hpipm_ocp_iter_solved[j,k] += 1
        # Solve time
        for j in range(MAX_QP_TIME):
            if('CSQP' in SOLVERS): 
                if(csqp_time_ik < j): csqp_time_solved[j,k] += 1
            if('OSQP' in SOLVERS): 
                if(osqp_time_ik < j): osqp_time_solved[j,k] += 1
            if('HPIPM_DENSE' in SOLVERS): 
                if(hpipm_dense_time_ik < j): hpipm_dense_time_solved[j,k] += 1
            if('HPIPM_OCP' in SOLVERS): 
                if(hpipm_ocp_time_ik < j): hpipm_ocp_time_solved[j,k] += 1

# Generate plot of number of iterations for each problem
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
 
# x-axis : max number of iterations
xdata     = range(0,MAX_QP_ITER)
for k in range(N_pb):
    fig0, ax0 = plt.subplots(1, 1, figsize=(19.2,10.8))
    if('CSQP' in SOLVERS): 
        ax0.plot(xdata, csqp_iter_solved[:,k]/N_samples, color='r', linestyle='solid', linewidth=4, label='CSQP') 
    if('OSQP' in SOLVERS): 
        ax0.plot(xdata, osqp_iter_solved[:,k]/N_samples, color='y', linestyle='dashed', linewidth=4, label='OSQP') 
    if('HPIPM_DENSE' in SOLVERS): 
        ax0.plot(xdata, hpipm_dense_iter_solved[:,k]/N_samples, color='g', linestyle='dotted',linewidth=4, label='HPIPM (dense)') 
    if('HPIPM_OCP' in SOLVERS): 
        ax0.plot(xdata, hpipm_ocp_iter_solved[:,k]/N_samples, color='b', linestyle='dashdot', linewidth=4, label='HPIPM (OCP)') #marker='o', markerfacecolor='b', linestyle='-', markersize=12, markeredgecolor='k', alpha=1., label='SQP')
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
for k in range(N_pb):
    fig0, ax0 = plt.subplots(1, 1, figsize=(19.2,10.8))
    if('CSQP' in SOLVERS): 
        ax0.plot(xdata, csqp_time_solved[:,k]/N_samples, color='r', linestyle='solid', linewidth=4, label='CSQP') 
    if('OSQP' in SOLVERS): 
        ax0.plot(xdata, osqp_time_solved[:,k]/N_samples, color='y', linestyle='dashed', linewidth=4, label='OSQP') 
    if('HPIPM_DENSE' in SOLVERS): 
        ax0.plot(xdata, hpipm_dense_time_solved[:,k]/N_samples, color='g', linestyle='dotted', linewidth=4, label='HPIPM (dense)') 
    if('HPIPM_OCP' in SOLVERS): 
        ax0.plot(xdata, hpipm_ocp_time_solved[:,k]/N_samples, color='b', linestyle='dashdot', linewidth=4, label='HPIPM (OCP)') #marker='o', markerfacecolor='b', linestyle='-', markersize=12, markeredgecolor='k', alpha=1., label='SQP')
    # Set axis and stuff
    ax0.set_ylabel('Percentage of problems solved', fontsize=26)
    ax0.set_xlabel('Max. solving time (ms)', fontsize=26)
    ax0.set_ylim(-0.02, 1.02)
    ax0.set_xscale("log")
    ax0.tick_params(axis = 'y', labelsize=22)
    ax0.tick_params(axis = 'x', labelsize=22)
    ax0.grid(True) 
    # Legend 
    handles0, labels0 = ax0.get_legend_handles_labels()
    fig0.legend(handles0, labels0, loc='lower right', bbox_to_anchor=(0.902, 0.1), prop={'size': 26}) 
    # Save, show , clean
    if(SAVE):
        fig0.savefig('/tmp/data_sqp_paper_croc2/bench_'+names[k]+'_SEED='+str(SEED)+'_MAXITER='+str(MAXITER)+'_TOL='+str(TOL)+'.pdf', bbox_inches="tight")




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


# def create_full_qp(problem, ndx, nu, gap, y):
#     '''
#     Creates the full QP (needed for benchmarks)
#     '''
#     P = np.zeros((problem.T*(ndx + nu), problem.T*(ndx + nu)))
#     q = np.zeros(problem.T*(ndx + nu))
#     Asize = problem.T*(ndx + nu)
#     A = np.zeros((problem.T*ndx, Asize))
#     B = np.zeros(problem.T*ndx)
#     for t, (model, data) in enumerate(zip(problem.runningModels, problem.runningDatas)):
#         index_u = problem.T*ndx + t * nu
#         if t>=1:
#             index_x = (t-1) * ndx
#             P[index_x:index_x+ndx, index_x:index_x+ndx] = data.Lxx.copy()
#             P[index_x:index_x+ndx, index_u:index_u+nu] = data.Lxu.copy()
#             P[index_u:index_u+nu, index_x:index_x+ndx] = data.Lxu.T.copy()
#             q[index_x:index_x+ndx] = data.Lx.copy()
#         P[index_u:index_u+nu, index_u:index_u+nu] = data.Luu.copy()
#         q[index_u:index_u+nu] = data.Lu.copy()
#         A[t * ndx: (t+1) * ndx, index_u:index_u+nu] = - data.Fu.copy() 
#         A[t * ndx: (t+1) * ndx, t * ndx: (t+1) * ndx] = np.eye(ndx)
#         if t >=1:
#             A[t * ndx: (t+1) * ndx, (t-1) * ndx: t * ndx] = - data.Fx.copy()
#         B[t * ndx: (t+1) * ndx] = gap[t].copy()
#     P[(problem.T-1)*ndx:problem.T*ndx, problem.T*ndx-ndx:problem.T*ndx] = problem.terminalData.Lxx.copy()
#     q[(problem.T-1)*ndx:problem.T*ndx] = problem.terminalData.Lx.copy()
#     n_in = sum([len(y[i]) for i in range(len(y))])
#     C = np.zeros((n_in, problem.T*(ndx + nu)))
#     l = np.zeros(n_in)
#     u = np.zeros(n_in)
#     nin_count = 0
#     index_x = problem.T*ndx
#     for t, (model, data) in enumerate(zip(problem.runningModels, problem.runningDatas)):
#         if model.ng == 0:
#             continue
#         l[nin_count: nin_count + model.ng] = model.g_lb - data.g
#         u[nin_count: nin_count + model.ng] = model.g_ub - data.g
#         if t > 0:
#             C[nin_count: nin_count + model.ng, (t-1)*ndx: t*ndx] = data.Gx
#         C[nin_count: nin_count + model.ng, index_x+t*nu: index_x+(t+1)*nu] = data.Gu
#         nin_count += model.ng
#     model = problem.terminalModel
#     data = problem.terminalData
#     if model.ng != 0:
#         l[nin_count: nin_count + model.ng] = model.g_lb - data.g
#         u[nin_count: nin_count + model.ng] = model.g_ub - data.g
#         C[nin_count: nin_count + model.ng, (problem.T-1)*ndx: problem.T*ndx] = data.Gx
#         nin_count += model.ng
#     return P, q, A, B, C, u, l
