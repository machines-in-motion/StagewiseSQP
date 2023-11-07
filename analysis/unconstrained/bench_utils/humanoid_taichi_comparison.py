import pathlib
import os
import sys
python_path = pathlib.Path('.').absolute().parent.parent/'python'
os.sys.path.insert(1, str(python_path))

import crocoddyl
import numpy as np
import example_robot_data
import pinocchio


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

# Problem definition & warm-start
x0 = np.concatenate([q0, pinocchio.utils.zero(state.nv)])
problem = crocoddyl.ShootingProblem(x0, [runningModel1] * T + [runningModel2] * T + [runningModel3] * T, terminalModel) 
xs = [x0] * (problem.T + 1)
us = problem.quasiStatic([x0] * problem.T)





# Set up solvers
MAXITER   = 500
TOL       = 1e-6
CALLBACKS = True
KKT_COND  = True

# SQP
# from sqp_ocp.solvers import CSSQPCPP,SQP

solverSQP = crocoddyl.SolverSQP(problem)
# solverSQP = CSSQPCPP(problem)
# solverSQP.termination_tolerance = TOL
# solverSQP.VERBOSE = CALLBACKS
solverSQP.termination_tolerance = TOL
solverSQP.with_callbacks = CALLBACKS
solverSQP.use_kkt_condition = KKT_COND
print("Solver SQP :")  
# print("  Use KKT condition     = ", solverSQP.use_kkt_condition)
print("  Termination tolerance = ", solverSQP.termination_tolerance)
print("  Penalty parameter mu  = ", solverSQP.mu)
print("  use_heuristic_line_search  = ", solverSQP.use_heuristic_line_search)


# FDDP 
solverFDDP = crocoddyl.SolverFDDP(problem)
solverFDDP.termination_tolerance = TOL
if(CALLBACKS): solverFDDP.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
solverFDDP.use_kkt_condition = KKT_COND
print("Solver FDPP :")  
print("  Use KKT condition     = ", solverFDDP.use_kkt_condition)
print("  Termination tolerance = ", solverFDDP.termination_tolerance)


# SOLVE SQP
mu_values      = [1e0, 1e2, 1e3]
converged_SQP = []
iter_SQP      = []
for mu in mu_values:
    solverSQP.mu = mu
    print('------------------')
    print("mu = ", solverSQP.mu)
    converged_SQP.append(solverSQP.solve(xs, us, MAXITER, False))
    print("nb iter = ", solverSQP.iter)
    iter_SQP.append(solverSQP.iter)
    print('------------------')
    
# SOLVE FDDP
converged_fddp = solverFDDP.solve(xs, us, MAXITER, False)
iter_fddp      = solverFDDP.iter

# Print
print("SQP mu \n", mu_values)
print("SQP iter \n", iter_SQP)
print("SQP converged \n", converged_SQP)
print("FDDP iter \n", iter_fddp)
print("FDDP converged \n", converged_fddp)