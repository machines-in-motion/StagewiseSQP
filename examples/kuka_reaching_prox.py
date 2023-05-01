### Kuka reaching example with different constraint implementation

import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'python'
os.sys.path.insert(1, str(python_path))

import crocoddyl
import numpy as np
import pinocchio as pin
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=4, linewidth=180)
import ocp_utils
from sqp_ocp.constraint_model import StateConstraintModel, ControlConstraintModel, EndEffConstraintModel, NoConstraint, ConstraintModelStack
from sqp_ocp.solvers import SQPOCP


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



# choose scenario: 0 or 1 or 2 or 3
option = 0

if option == 0:    
  clip_state_max = np.array([np.inf]*14)
  clip_state_min = -np.array([np.inf]*7 + [0.5]*7)
  clip_ctrl = np.array([np.inf, 40 , np.inf, np.inf, np.inf, np.inf , np.inf] )


  statemodel = StateConstraintModel(clip_state_min, clip_state_max, 14, 14, 7)
  controlmodel = ControlConstraintModel(-clip_ctrl, clip_ctrl, 7, 14, 7)

  ConstraintModel = ConstraintModelStack([statemodel, controlmodel], 14, 7)

  clip_state_end = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf , np.inf] + [0.001]*7)
  TerminalConstraintModel = StateConstraintModel(-clip_state_end, clip_state_end, 14, 14, 7)
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


xs = [x0] * (T+1)
us = [np.zeros(nu)] * T 
ddp_prox = SQPOCP(problem, constraintModels, "ProxQP")
ddp = crocoddyl.SolverPROXQP(problem, constraintModels)
ddpf = crocoddyl.SolverFADMM(problem, constraintModels)

verbose = True
tol = 1e-2
ddp.with_callbacks  = verbose
ddpf.with_callbacks  = verbose

ddp.use_heuristic_ls = True
ddp_prox.verbose = verbose
ddp_prox.verboseQP = False
ddp.termination_tolerance = tol
ddpf.termination_tolerance = tol

ddpf.max_qp_iters = 10000

iter = 10
# ddp.calc(True)
# ddp.computeDirect

import time

t1 = time.time()
ddp.solve(xs, us, iter)
t2 = time.time()
print(t2 - t1)

t1 = time.time()
ddpf.solve(xs, us, iter)
t2 = time.time()
print(t2 - t1)


# t1 = time.time()
# ddp_prox.solve(xs, us, 6)
# t2 = time.time()
# print(t2 - t1)



# print(np.linalg.norm(ddp.P - ddp_prox.P))
# print(np.linalg.norm(ddp.A - ddp_prox.A))
# print(np.linalg.norm(ddp.C - ddp_prox.C))
# print(np.linalg.norm(ddp.q - ddp_prox.q))
# print(np.linalg.norm(ddp.b - ddp_prox.b))

# print(np.linalg.norm(np.array(ddp.dx) - np.array(ddp_prox.dx)))
# print(np.linalg.norm(np.array(ddp.du) - np.array(ddp_prox.du)))


# print(np.linalg.norm(np.array(ddp.xs) - np.array(ddp_prox.xs)))
# print(np.linalg.norm(np.array(ddp.us) - np.array(ddp_prox.us)))

# # print(np.linalg.norm(ddp.l - ddp_prox.l))
# # print(np.linalg.norm(ddp.u - ddp_prox.u))

# y1 = np.concatenate(ddp.y).flatten()
# y2 = np.concatenate(ddp_prox.y).flatten()
# print(np.linalg.norm(y1 - y2))


# lag1 = np.concatenate(ddp.lag_mul).flatten()
# lag2 = np.concatenate(ddp_prox.lag_mul).flatten()
# print(np.linalg.norm(lag1 - lag2))

# import time

# t1 = time.time()
# ddp.solve(xs, us, 10)
# t2 = time.time()
# print(t2 - t1)
# print(ddp.time)
# Extract DDP data and plot

# ddp_data = ocp_utils.extract_ocp_data(ddp, ee_frame_name='contact')

# ocp_utils.plot_ocp_results(ddp_data, which_plots="all", labels=None, markers=['.'], colors=['b'], sampling_plot=1, SHOW=True)

