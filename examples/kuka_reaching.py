### Kuka reaching example with different constraint implementation

import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'python'
os.sys.path.insert(1, str(python_path))

import crocoddyl
import numpy as np
import pinocchio as pin
np.set_printoptions(precision=4, linewidth=180)
import ocp_utils
from sqp_ocp.constraint_model import StateConstraintModel, EndEffConstraintModel, NoConstraintModel
from sqp_ocp.solvers import CSSQP


# # # # # # # # # # # # #
### LOAD ROBOT MODEL  ###
# # # # # # # # # # # # #

# Or use robot_properties_kuka 
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

# # # # # # # # # # # # # # #
###  SETUP CROCODDYL OCP  ###
# # # # # # # # # # # # # # #

# State and actuation model
state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFull(state)

# Running and terminal cost models
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)


# Create cost terms 
  # Control regularization cost
uResidual = crocoddyl.ResidualModelControlGrav(state)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
  # State regularization cost
xResidual = crocoddyl.ResidualModelState(state, x0)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
  # endeff frame translation cost
endeff_frame_id = model.getFrameId("contact")
endeff_translation = np.array([0.7, 0, 1.1]) # move endeff +30 cm along x in WORLD frame
frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, endeff_frame_id, endeff_translation)
frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)


# Add costs
runningCostModel.addCost("stateReg", xRegCost, 1e-1)
runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
runningCostModel.addCost("translation", frameTranslationCost, 1)
terminalCostModel.addCost("stateReg", xRegCost, 1e-1)
terminalCostModel.addCost("translation", frameTranslationCost, 1)

# Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel)
terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel)

# Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
dt = 5e-2
runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)

# Create the shooting problem
T = 40
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)



# choose scenario: 0 or 1 or 2 or 3
option = 0

if option == 0:    
  clip_state_max = np.array([np.inf]*14)
  clip_state_min = -np.array([np.inf]*7 + [0.5]*7)
  clip_ctrl = np.array([np.inf, 40 , np.inf, np.inf, np.inf, np.inf , np.inf] )


  statemodel = crocoddyl.StateConstraintModel(state, 7, clip_state_min, clip_state_max, 'stateConstraint')
  controlmodel = crocoddyl.ControlConstraintModel(state, 7,  -clip_ctrl, clip_ctrl, 'ctrlConstraint')

  nc = statemodel.nc + controlmodel.nc
  ConstraintModel = crocoddyl.ConstraintStack([statemodel, controlmodel], state, nc, 7, 'runningConstraint')
  clip_state_end = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf , np.inf] + [0.01]*7)
  terminal = crocoddyl.StateConstraintModel(state, 7, clip_state_min, clip_state_max, 'stateConstraint')
  TerminalConstraintModel = crocoddyl.ConstraintStack([terminal], state, 14, 7, 'terminalConstraint')

  constraintModels =[controlmodel] * (T) + [terminal]
elif option == 1:    
  clip_state_max = np.array([np.inf]*14)
  clip_state_min = -np.array([np.inf]*7 + [0.5]*7)
  statemodel = StateConstraintModel(state, 7, clip_state_min, clip_state_max, 'stateConstraint')
  clip_state_end = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf , np.inf] + [0.00]*7)
  TerminalConstraintModel = StateConstraintModel(state, 7, -clip_state_end, clip_state_end, 'CtrlConstraint')
  constraintModels =  [NoConstraintModel(state, 7, "none")] + [statemodel] * (T-1) + [TerminalConstraintModel]

elif option == 2:
  endeff_translation = np.array([0.7, 0, 1.1]) # move endeff +30 cm along x in WORLD frame

  lmin = np.array([-np.inf, endeff_translation[1], endeff_translation[2]])
  lmax =  np.array([np.inf, endeff_translation[1], endeff_translation[2]])
  constraintModels = [NoConstraintModel(state, 7, "none")] + [EndEffConstraintModel(state, 7, fid, lmin, lmax, "ee")] * T


elif option == 3:
  constraintModels = [NoConstraintModel(state, 7, "none")] * (T+1)


xs = [x0] * (T+1)
us = [np.zeros(nu)] * T 



qp_iters = 1000
sqp_ites = 10
eps_abs = 1e-4
eps_rel = 1e-4
termination_tol = 1e-4


ddppy = CSSQP(problem, constraintModels, "StagewiseQP")
ddppy.eps_abs = eps_abs
ddppy.eps_rel = eps_rel
ddppy.termination_tol = termination_tol
ddppy.verbose = True
ddppy.solve(xs, us, qp_iters)

# Extract DDP data and plot
ddp_data = ocp_utils.extract_ocp_data(ddppy, ee_frame_name="contact")
ocp_utils.plot_ocp_results(ddp_data, which_plots="all", labels=None, markers=['.'], colors=['b'], sampling_plot=1, SHOW=True)

