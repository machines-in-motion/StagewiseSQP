### UNIT TEST for GNMS solvers on Kuka reaching task

import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'python'
os.sys.path.insert(1, str(python_path))

import crocoddyl
import numpy as np
import pinocchio as pin
np.set_printoptions(precision=4, linewidth=180)
from sqp_ocp.solvers import GNMS, GNMSCPP

LINE_WIDTH = 100

# # # # # # # # # # # # #
### LOAD ROBOT MODEL  ###
# # # # # # # # # # # # #

# Or use robot_properties_kuka 
from robot_properties_kuka.config import IiwaConfig
robot = IiwaConfig.buildRobotWrapper()

model = robot.model
nq = model.nq; nv = model.nv; nu = nq; nx = nq+nv
q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.])
v0 = np.zeros(nv)
x0 = np.concatenate([q0, v0]).copy()
robot.framesForwardKinematics(q0)
robot.computeJointJacobians(q0)


# # # # # # # # # # # # # # #
###  SETUP CROCODDYL OCP  ###
# # # # # # # # # # # # # # #

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
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

print("TEST KUKA PROBLEM GNMS".center(LINE_WIDTH, "-"))

xs = [x0] * (T+1)
us = [np.zeros(nu)] * T 

# Create solvers
ddp0 = GNMS(problem)
ddp1 = GNMSCPP(problem, use_heuristic_ls=True, VERBOSE=True)
ddp2 = crocoddyl.SolverGNMS(problem)
ddp3 = crocoddyl.SolverFDDP(problem)

ddp0.solve(xs, us, 100)
ddp1.solve(xs, us, 100)
ddp2.solve(xs, us, 100)
ddp3.solve(xs, us, 100)


##### UNIT TEST #####################################

set_tol = 1e-6
assert np.linalg.norm(np.array(ddp0.xs) - np.array(ddp1.xs)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp0.us) - np.array(ddp1.us)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp0.xs) - np.array(ddp2.xs)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp0.us) - np.array(ddp2.us)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp0.xs) - np.array(ddp3.xs)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp0.us) - np.array(ddp3.us)) < set_tol, "Test failed"

assert ddp0.cost - ddp1.cost < set_tol, "Test failed"
assert ddp0.cost - ddp2.cost < set_tol, "Test failed"
assert ddp0.cost - ddp3.cost < set_tol, "Test failed"

assert np.linalg.norm(np.array(ddp0.lag_mul) - np.array(ddp1.lag_mul)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp0.lag_mul) - np.array(ddp2.lag_mul)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp0.lag_mul) - np.array(ddp3.lag_mul)) < set_tol, "Test failed"

assert ddp0.KKT - ddp1.KKT < set_tol, "Test failed"
assert ddp0.KKT - ddp2.KKT < set_tol, "Test failed"
assert ddp0.KKT - ddp3.KKT < set_tol, "Test failed"

print("TEST PASSED".center(LINE_WIDTH, "-"))
print("\n")