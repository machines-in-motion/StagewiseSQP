### Kuka reaching example with different constraint implementation

import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'python'
os.sys.path.insert(1, str(python_path))

import crocoddyl
import numpy as np
import pinocchio as pin
np.set_printoptions(precision=4, linewidth=180)
from sqp_ocp.constraint_model import NoConstraintModel, StateConstraintModel, ControlConstraintModel, EndEffConstraintModel, ConstraintModelStack
from sqp_ocp.solvers import CSSQP

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
fid = model.getFrameId("contact")

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
uResidual = crocoddyl.ResidualModelContactControlGrav(state)
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
running_DAM = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, crocoddyl.ContactModelMultiple(state, actuation.nu), runningCostModel)
terminal_DAM = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, crocoddyl.ContactModelMultiple(state, actuation.nu), terminalCostModel)

# Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
dt = 1e-2
runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)

# Create the shooting problem
T = 10
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)


# choose scenario: 0 or 1 or 2 or 3
option = 0

if option == 0:    
  clip_state_max = np.array([np.inf]*14)
  clip_state_min = -np.array([np.inf]*7 + [0.5]*7)
  clip_ctrl = np.array([np.inf, 40 , np.inf, np.inf, np.inf, np.inf , np.inf] )


  statemodel = StateConstraintModel(state, 7, clip_state_min, clip_state_max, "state")
  controlmodel = ControlConstraintModel(state, 7,  -clip_ctrl, clip_ctrl, "control")

  nc = statemodel.nc + controlmodel.nc
  ConstraintModel = ConstraintModelStack([statemodel, controlmodel], state, nc, 7, "stack")
  clip_state_end = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf , np.inf] + [0.01]*7)
  terminal = StateConstraintModel(state, 7, clip_state_min, clip_state_max, "state")
  TerminalConstraintModel = ConstraintModelStack([terminal], state, 14, 7, "stack")

  constraintModels =[controlmodel] * (T) + [terminal]
elif option == 1:    
  clip_state_max = np.array([np.inf]*14)
  clip_state_min = -np.array([np.inf]*7 + [0.5]*7)
  statemodel = StateConstraintModel(state, 7, clip_state_min, clip_state_max, "state")
  clip_state_end = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf , np.inf] + [0.00]*7)
  TerminalConstraintModel = StateConstraintModel(state, 7, -clip_state_end, clip_state_end, "state")
  constraintModels =  [NoConstraintModel(state, 7, "none")] + [statemodel] * (T-1) + [TerminalConstraintModel]

elif option == 2:
  endeff_translation = np.array([0.7, 0, 1.1]) # move endeff +30 cm along x in WORLD frame

  lmin = np.array([-np.inf, endeff_translation[1], endeff_translation[2]])
  lmax =  np.array([np.inf, endeff_translation[1], endeff_translation[2]])
  constraintModels = [NoConstraintModel(state, 7, "none")] + [EndEffConstraintModel(state, 7, fid, lmin, lmax, "ee")] * T


elif option == 3:
  constraintModels = [NoConstraintModel(state, 7, "none")] * (T+1)


xs = [x0] * (T+1)
us = [np.zeros(nu)] * T 


ddp = crocoddyl.SolverFADMM(problem, constraintModels)

qp_iters = 1000
sqp_ites = 10
eps_abs = 1e-4
eps_rel = 1e-4

ddp.with_callbacks = False
ddp.use_filter_ls = True
ddp.filter_size = 10
ddp.termination_tolerance = 1e-3
ddp.warm_start = True
ddp.max_qp_iters = qp_iters
ddp.eps_abs = eps_abs
ddp.eps_rel = eps_rel



ddp.solve(xs, us, sqp_ites)

ddp.calc(True)
ddp.backwardPass_without_constraints()
ddp.forwardPass()
ddp.update_lagrangian_parameters(False)
ddp.backwardPass()


xs = [x0] * (T+1)
us = [np.zeros(nu)] * T 


problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

ddppy = CSSQP(problem, constraintModels, "StagewiseQP")
ddppy.eps_abs = eps_abs
ddppy.eps_rel = eps_rel

ddppy.verbose = False
# ddppy.verboseQP = False
ddppy.max_iters = qp_iters

ddppy.solve(xs, us, sqp_ites)

ddppy.calc(True)
ddppy.backwardPass_without_constraints()
ddppy.computeUpdates()
ddppy.update_lagrangian_parameters_infinity(False)

ddppy.backwardPass()


############## UNIT TEST #################################
tol = 1e-3

for i in range(0, T):
  print (ddp.k[i] + ddppy.l[i])
  assert (np.linalg.norm(ddp.k[i] + ddppy.l[i])) < tol
  assert(np.linalg.norm(ddp.K[i] + ddppy.L[i])) < tol


# # print("_____________________________________________")
for i in range(T, 0, -1):
  assert (np.linalg.norm(ddp.Vxx[i] -  ddppy.S[i])) < tol
  assert(np.linalg.norm(ddp.Vx[i] -  ddppy.s[i])) < tol

assert (np.linalg.norm(np.array(ddp.dx_tilde) - ddppy.dx_tilde)) < tol
assert (np.linalg.norm(np.array(ddp.du_tilde) - ddppy.du_tilde)) < tol


assert (ddp.rho_sparse - ddppy.rho_sparse) < tol
assert (ddp.norm_primal - ddppy.norm_primal) < tol
assert (ddp.norm_dual - ddppy.norm_dual) < tol
assert (ddp.norm_primal_rel - ddppy.norm_primal_rel) < tol
assert (ddp.norm_dual_rel - ddppy.norm_dual_rel) < tol

for i in range(len(ddppy.lag_mul)):
  assert(np.linalg.norm((ddp.lag_mul)[i] - ddppy.lag_mul[i])) < tol
  assert(np.linalg.norm((ddp.y)[i] - ddppy.y[i])) < tol
  assert (np.linalg.norm((ddp.get_rho_vec)[i] - ddppy.rho_vec[i])) < tol
  assert (np.linalg.norm((ddp.z)[i] - ddppy.z[i])) < tol

assert (np.linalg.norm(np.array(ddp.xs) - ddppy.xs)/(T+1)) < tol
assert (np.linalg.norm(np.array(ddp.us) - ddppy.us)/T) < tol


print("TEST PASSED".center(LINE_WIDTH, "-"))
print("\n")