import crocoddyl
import numpy as np
import mim_solvers
import ocp_utils


# # # # # # # # # # # # #
### LOAD ROBOT MODEL  ###
# # # # # # # # # # # # #

# Or use robot_properties_kuka
from robot_properties_kuka.config import IiwaConfig

robot = IiwaConfig.buildRobotWrapper()

model = robot.model
nq = model.nq
nv = model.nv
nu = nq
nx = nq + nv
q0 = np.array([0.1, 0.7, 0.0, 0.7, -0.5, 1.5, 0.0])
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


# Create cost terms
# Control regularization cost
uResidual = crocoddyl.ResidualModelControlGrav(state)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
# State regularization cost
xResidual = crocoddyl.ResidualModelState(state, x0)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
# endeff frame translation cost
endeff_frame_id = model.getFrameId("contact")
endeff_translation = np.array([0.7, -0.02, 1.1])
frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
    state, endeff_frame_id, endeff_translation
)
frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)

# Create contraints
# Control box constraint
uResidual = crocoddyl.ResidualModelControl(state)
control_lim = np.array([100, 50, 100, 80, 80, 60, 60])
controlbound_contraint = crocoddyl.ConstraintModelResidual(
    state, uResidual, -control_lim, control_lim
)
# End effector path contraint on y.
frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
    state, endeff_frame_id, np.zeros(3)
)
ee_contraint = crocoddyl.ConstraintModelResidual(
    state,
    frameTranslationResidual,
    np.array([-1.0, -0.02, -1]),
    np.array([1.0, -0.02, 2]),
)

# Create the running models
runningModels = []
dt = 5e-2
T = 20
for t in range(T + 1):
    runningCostModel = crocoddyl.CostModelSum(state)
    # Add costs
    runningCostModel.addCost("stateReg", xRegCost, 1e-1)
    runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
    if t != T:
        runningCostModel.addCost("translation", frameTranslationCost, 4)
    else:
        runningCostModel.addCost("translation", frameTranslationCost, 40)
    # Define contraints
    constraints = crocoddyl.ConstraintModelManager(state, nu)
    if t != 0:
        constraints.addConstraint("ee_bound", ee_contraint)
    if t != T:
        constraints.addConstraint("u_bound", controlbound_contraint)
    # Create Differential action model
    running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, runningCostModel, constraints
    )
    # Apply Euler integration
    running_model = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
    runningModels.append(running_model)


# Create the shooting problem
problem = crocoddyl.ShootingProblem(x0, runningModels[:-1], runningModels[-1])


# # # # # # # # # # # # #
###     SOLVE OCP     ###
# # # # # # # # # # # # #

# Define warm start
xs = [x0] * (T + 1)
us = [np.zeros(nu)] * T

# Define solver
solver = mim_solvers.SolverCSQP(problem)
max_iter = 100
solver.termination_tolerance = 1e-2
solver.with_callbacks = True
solver.use_filter_line_search = False
solver.filter_size = max_iter

# Solve
solver.solve(xs, us, max_iter)


# Extract DDP data and plot
data = ocp_utils.extract_ocp_data(solver, ee_frame_name="contact")
ocp_utils.plot_ocp_results(
    data,
    which_plots="all",
    labels=None,
    markers=["."],
    colors=["b"],
    sampling_plot=1,
    SHOW=True,
)
