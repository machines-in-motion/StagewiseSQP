import crocoddyl
import pinocchio as pin
import numpy as np
import mim_solvers
import pin_utils, ocp_utils

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
x0 = np.concatenate([q0, v0])
robot.framesForwardKinematics(q0)
robot.computeJointJacobians(q0)


# # # # # # # # # # # # # # #
###  SETUP CROCODDYL OCP  ###
# # # # # # # # # # # # # # #

# State and actuation model
state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFull(state)

# Running and terminal cost models
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)

# Contact model 
contactModel = crocoddyl.ContactModelMultiple(state, actuation.nu)

# Create 3D contact on the en-effector frame
contact_frame_id = model.getFrameId("contact")
contact_position = robot.data.oMf[contact_frame_id].copy()
baumgarte_gains  = np.array([0., 50.])
pinRef           = pin.LOCAL_WORLD_ALIGNED
contact3d = crocoddyl.ContactModel6D(state, contact_frame_id, contact_position, pinRef, nu, baumgarte_gains) 

# Populate contact model with contacts
contactModel.addContact("contact", contact3d, active=True)


# Create cost terms 
  # Control regularization cost
uResidual = crocoddyl.ResidualModelContactControlGrav(state)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
  # State regularization cost
xResidual = crocoddyl.ResidualModelState(state, x0)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
  # End-effector frame force cost
desired_wrench = np.array([20., 0., -100., 0., 0., 0.])
frameForceResidual = crocoddyl.ResidualModelContactForce(state, contact_frame_id, pin.Force(desired_wrench), 6, actuation.nu)
contactForceCost = crocoddyl.CostModelResidual(state, frameForceResidual)

# Populate cost models with cost terms
runningCostModel.addCost("stateReg", xRegCost, 1e-2)
runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
runningCostModel.addCost("force", contactForceCost, 10.)
terminalCostModel.addCost("stateReg", xRegCost, 1e-2)

# Constraint model
Fmin = np.array([0., -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
Fmax = np.array([0., np.inf, np.inf, np.inf, np.inf, np.inf])

forceResidual = crocoddyl.ResidualModelContactForce(state, contact_frame_id, pin.Force(np.zeros(6)), 6, actuation.nu)
forcebound_contraint = crocoddyl.ConstraintModelResidual(
    state, forceResidual, Fmin, Fmax
)
constraints = crocoddyl.ConstraintModelManager(state, nu)
constraints.addConstraint("force_constraint", forcebound_contraint)
terminal_constraint = crocoddyl.ConstraintModelManager(state, nu)

# Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
running_DAM = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, runningCostModel, constraints, inv_damping=0., enable_force=True)
terminal_DAM = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, terminalCostModel, terminal_constraint, inv_damping=0., enable_force=True)

# Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
dt = 1e-2
runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)

# Create the shooting problem
T = 10
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)





# Define Solver
solver = mim_solvers.SolverCSQP(problem)
solver.termination_tolerance = 1e-2
solver.with_callbacks = True

# Warm start
xs = [x0 for i in range(T+1)]
us = solver.problem.quasiStatic(xs[:-1])

# Solve
sqp_ites = 10
solver.solve(xs, us, sqp_ites)


# Extract DDP data and plot
ddp_data = {}
ddp_data = ocp_utils.extract_ocp_data(solver, ee_frame_name='contact', ct_frame_name='contact')
ocp_utils.plot_ocp_results(ddp_data, which_plots='all', labels=None, markers=['.'], colors=['b'], sampling_plot=1, SHOW=True)
