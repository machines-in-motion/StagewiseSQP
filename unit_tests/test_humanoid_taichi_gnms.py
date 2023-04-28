import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'python'
os.sys.path.insert(1, str(python_path))
import sys
import crocoddyl
import numpy as np
import example_robot_data
import pinocchio
np.set_printoptions(precision=4, linewidth=180)
from sqp_ocp.solvers import GNMS, GNMSCPP

LINE_WIDTH = 100

# Load robot
robot  = example_robot_data.load('talos')
rmodel = robot.model
lims   = rmodel.effortLimit
rmodel.effortLimit = lims

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
pinocchio.forwardKinematics(rmodel, rdata, q0)
pinocchio.updateFramePlacements(rmodel, rdata)
rfPos0 = rdata.oMf[rightFootId].translation
lfPos0 = rdata.oMf[leftFootId].translation
refGripper = rdata.oMf[rmodel.getFrameId("gripper_left_joint")].translation
comRef = (rfPos0 + lfPos0) / 2
comRef[2] = pinocchio.centerOfMass(rmodel, rdata, q0)[2].item()

# Create two contact models used along the motion
contactModel1Foot = crocoddyl.ContactModelMultiple(state, actuation.nu)
contactModel2Feet = crocoddyl.ContactModelMultiple(state, actuation.nu)
supportContactModelLeft = crocoddyl.ContactModel6D(state, leftFootId, pinocchio.SE3.Identity(), actuation.nu,
                                                   np.array([0, 40]))
supportContactModelRight = crocoddyl.ContactModel6D(state, rightFootId, pinocchio.SE3.Identity(), actuation.nu,
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
handTrackingResidual = crocoddyl.ResidualModelFramePlacement(state, endEffectorId, pinocchio.SE3(np.eye(3), target),
                                                             actuation.nu)
handTrackingActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1] * 3 + [0.0001] * 3)**2)
handTrackingCost = crocoddyl.CostModelResidual(state, handTrackingActivation, handTrackingResidual)

footTrackingResidual = crocoddyl.ResidualModelFramePlacement(state, leftFootId,
                                                             pinocchio.SE3(np.eye(3), np.array([0., 0.4, 0.])),
                                                             actuation.nu)
footTrackingActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1, 1, 0.1] + [1.] * 3)**2)
footTrackingCost1 = crocoddyl.CostModelResidual(state, footTrackingActivation, footTrackingResidual)
footTrackingResidual = crocoddyl.ResidualModelFramePlacement(state, leftFootId,
                                                             pinocchio.SE3(np.eye(3), np.array([0.3, 0.15, 0.35])),
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

# Then let's added the running and terminal cost functions
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
x0 = np.concatenate([q0, pinocchio.utils.zero(state.nv)])
problem = crocoddyl.ShootingProblem(x0, [runningModel1] * T + [runningModel2] * T + [runningModel3] * T, terminalModel)

print("TEST HUMANOID TAICHI PROBLEM GNMS".center(LINE_WIDTH, "-"))

# Warm-start 
# Solving it with the DDP algorithm
xs = [x0] * (problem.T + 1)
us = problem.quasiStatic([x0] * problem.T)

# Create solvers
ddp0 = GNMSCPP(problem)
ddp1 = crocoddyl.SolverGNMS(problem)

# Set Heuristic Line Search (filter)
ddp0.use_filter_ls = True
ddp1.use_filter_line_search = True

# Set callbacks
ddp0.VERBOSE = True
ddp1.with_callbacks = True

# Set tolerance 
ddp0.termination_tol = 1e-8
ddp1.termination_tol = 1e-8

# Set filter size
ddp0.filter_size = 10
ddp1.filter_size = 10
# ddp0.mu = 1e3
# ddp1.mu = 1e3

# ddp1.termination_tolerance = 1e-2
ddp0.solve(xs, us, 100, False)
ddp1.solve(xs, us, 100, False)

# ##### UNIT TEST #####################################

set_tol = 1e-6
assert np.linalg.norm(np.array(ddp0.xs) - np.array(ddp1.xs)) < set_tol, "Test failed"
# assert np.linalg.norm(np.array(ddp0.us) - np.array(ddp2.us)) < set_tol, "Test failed"

assert ddp0.cost - ddp1.cost < set_tol, "Test failed"
# assert ddp0.cost - ddp2.cost < set_tol, "Test failed"

assert np.linalg.norm(np.array(ddp0.lag_mul) - np.array(ddp1.lag_mul)) < set_tol, "Test failed"
# assert np.linalg.norm(np.array(ddp0.lag_mul) - np.array(ddp2.lag_mul)) < set_tol, "Test failed"

assert ddp0.KKT - ddp1.KKT < set_tol, "Test failed"
# assert ddp0.KKT - ddp2.KKT < set_tol, "Test failed"

print("TEST PASSED".center(LINE_WIDTH, "-"))
print("\n")