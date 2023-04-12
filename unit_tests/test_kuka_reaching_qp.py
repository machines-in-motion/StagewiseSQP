## UNIT TEST - to compare one QP solve of FADMM and FAdmmKKT for the kuka reaching problem
import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'python'
os.sys.path.insert(1, str(python_path))

import crocoddyl
import numpy as np
import pinocchio as pin
np.set_printoptions(precision=4, linewidth=180)
# import ocp_utils
from sqp_ocp.constraint_model import StateConstraintModel, ControlConstraintModel, EndEffConstraintModel, NoConstraint, ConstraintModelStack
from sqp_ocp.solvers import FADMM, QPSolvers


# # # # # # # # # # # # #
### LOAD ROBOT MODEL  ###
# # # # # # # # # # # # #
LINE_WIDTH = 100

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



# choose scenario: 0 or 1 or 2 or 3
option = 0

if option == 0:    
  clip_state_max = np.array([np.inf]*14)
  clip_state_min = -np.array([np.inf]*7 + [0.5]*7)
  clip_ctrl = np.array([np.inf, 40 , np.inf, np.inf, np.inf, np.inf , np.inf] )


  statemodel = StateConstraintModel(clip_state_min, clip_state_max, 7, 14, 7)
  controlmodel = ControlConstraintModel(-clip_ctrl, clip_ctrl, 7, 14, 7)

  ConstraintModel = ConstraintModelStack([statemodel, controlmodel], 14, 7)

  clip_state_end = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf , np.inf] + [0.001]*7)
  TerminalConstraintModel = StateConstraintModel(-clip_state_end, clip_state_end, 7, 14, 7)
  constraintModels = [controlmodel] + [controlmodel] * (T-1) + [TerminalConstraintModel]

elif option == 1:    
  clip_state_max = np.array([np.inf]*14)
  clip_state_min = -np.array([np.inf]*7 + [0.5]*7)
  statemodel = StateConstraintModel(clip_state_min, clip_state_max, 7, 14, 7)
  clip_state_end = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf , np.inf] + [0.001]*7)
  TerminalConstraintModel = StateConstraintModel(-clip_state_end, clip_state_end, 7, 14, 7)
  constraintModels =  [NoConstraint(14, 7)] + [statemodel] * (T-1) + [TerminalConstraintModel]

elif option == 2:
  lmin = np.array([-np.inf, endeff_translation[1], endeff_translation[2]])
  lmax =  np.array([np.inf, endeff_translation[1], endeff_translation[2]])
  constraintModels = [NoConstraint(14, 7)] + [EndEffConstraintModel(robot, lmin, lmax, 3, 14, 7)] * T


elif option == 3:
  constraintModels = [NoConstraint(14, 7)] * (T+1)


print("TEST KUKA PROBLEM : FADMM = FAdmmKKT".center(LINE_WIDTH, "-"))

xs = [x0] * (T+1)
us = [np.zeros(nu)] * T 
ddp1 = FADMM(problem, constraintModels, verbose = False)
ddp1.solve(xs, us, 1)

ddp2 = QPSolvers(problem, constraintModels, "FAdmmKKT", verbose = False)
ddp2.solve(xs, us, 1)

##### UNIT TEST #####################################

set_tol = 1e-6
assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp2.xs)) < set_tol, "Test failed"

assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp2.us)) < set_tol, "Test failed"


d_relaxed = np.concatenate(ddp1.dz_relaxed).flatten()
assert np.linalg.norm(d_relaxed - np.array(ddp2.x_k_1)) < set_tol, "Test failed"

z = np.concatenate(ddp1.z).flatten()
assert np.linalg.norm(z - np.array(ddp2.z_k)) < set_tol, "Test failed"

y = np.concatenate(ddp1.y).flatten()
assert np.linalg.norm(y - np.array(ddp2.y_k)) < set_tol, "Test failed"

assert np.linalg.norm(ddp1.norm_primal - np.array(ddp2.r_prim)) < set_tol, "Test failed"
assert np.linalg.norm(ddp1.norm_dual - np.array(ddp2.r_dual)) < set_tol, "Test failed"

assert np.linalg.norm(ddp1.norm_primal_rel - np.array(ddp2.eps_rel_prim)) < set_tol, "Test failed"
assert np.linalg.norm(ddp1.norm_dual_rel - np.array(ddp2.eps_rel_dual)) < set_tol, "Test failed"

assert np.linalg.norm(ddp1.rho_sparse - np.array(ddp2.rho_boyd)) < set_tol, "Test failed"

assert np.linalg.norm(ddp1.rho_estimate_sparse - np.array(ddp2.rho_estimate_boyd)) < set_tol, "Test failed"

rho = np.concatenate(ddp1.rho_vec).flatten()

assert np.linalg.norm(rho - np.array(ddp2.rho_vec_boyd)) < set_tol, "Test failed"

print("TESTS PASSED".center(LINE_WIDTH, "-"))
print("\n")
