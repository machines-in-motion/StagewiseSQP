### Kuka reaching example with different constraint implementation

import pathlib
import os
python_path = pathlib.Path('.').absolute().parent.parent/'python'
os.sys.path.insert(1, str(python_path))
import time
import crocoddyl
import numpy as np
import pinocchio as pin
np.set_printoptions(precision=4, linewidth=180)


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

# Optionally add armature to take into account actuator's inertia
# runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
# terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])

# Create the shooting problem
T = 10
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)


# Set up solvers
MAXITER   = 1
TOL       = 1e-8
CALLBACKS = False
KKT_COND  = True

xs = [x0] * (T+1)
us = [np.zeros(nu)] * T 
solverGNMS = crocoddyl.SolverGNMS(problem)
solverGNMS.termination_tolerance = TOL
solverGNMS.VERBOSE = CALLBACKS
# solverGNMS.use_heuristic_line_search = True
# solverGNMS.termination_tolerance = TOL
solverGNMS.with_callbacks = CALLBACKS
solverGNMS.use_kkt_condition = KKT_COND
solverGNMS.computeDirection(True)
t1 = time.time()
solverGNMS.computeDirection(False)

# tmp = solverGNMS.solve(xs, us, MAXITER, False)
t2 = time.time()
print(t2 - t1)

from sqp_ocp.constraint_model import StateConstraintModel, ControlConstraintModel, EndEffConstraintModel, NoConstraint, ConstraintModelStack
from sqp_ocp.solvers import SQPOCP

constraintModels = [NoConstraint(len(x0), actuation.nu)] * (T+1)

problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

solver = SQPOCP(problem, constraintModels, "ProxQP")
solver.verbose = True
tmp = solver.solve(xs, us, MAXITER, CALLBACKS)
print(solver.time)
# mu_values      = [1e-6, 1e-2, 1e-3]
# converged_gnms = []
# iter_gnms      = []
# for mu in mu_values:
#     solverGNMS.mu = mu
#     print('------------------')
#     print("mu = ", solverGNMS.mu)
#     t1 = time.time()
#     tmp = solverGNMS.solve(xs, us, MAXITER, False)
#     t2 = time.time()
#     converged_gnms.append(tmp)
#     print("GNMS", 1e3*(t2 - t1))
#     print("nb iter = ", solverGNMS.iter)

#     iter_gnms.append(solverGNMS.iter)
#     print('------------------')
    

# # SOLVE FDDP
# solverFDDP = crocoddyl.SolverFDDP(problem)
# solverFDDP.termination_tolerance = TOL
# if(CALLBACKS): 
#     solverFDDP.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

# t1 = time.time()
# converged_fddp = solverFDDP.solve(xs, us, MAXITER, False)
# t2 = time.time()
# print("FDDP", 1e3*(t2 - t1))

# iter_fddp      = solverFDDP.iter

# # Print
# print("GNMS mu \n", mu_values)
# print("GNMS iter \n", iter_gnms)
# print("GNMS converged \n", converged_gnms)
# print("FDDP iter \n", iter_fddp)
# print("FDDP converged \n", converged_fddp)
