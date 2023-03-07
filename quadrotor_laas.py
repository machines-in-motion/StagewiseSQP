import os
import sys

import crocoddyl
import pinocchio
import numpy as np
import example_robot_data
from gnms import GNMS
from gnms_cpp import GNMSCPP

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

hector = example_robot_data.load('hector')
robot_model = hector.model

target_pos = np.array([1., 0., 1.])
target_quat = pinocchio.Quaternion(1., 0., 0., 0.)

state = crocoddyl.StateMultibody(robot_model)

d_cog, cf, cm, u_lim, l_lim = 0.1525, 6.6e-5, 1e-6, 5., 0.1
tau_f = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.], [1., 1., 1., 1.], [0., d_cog, 0., -d_cog],
                  [-d_cog, 0., d_cog, 0.], [-cm / cf, cm / cf, -cm / cf, cm / cf]])
actuation = crocoddyl.ActuationModelMultiCopterBase(state, tau_f)

nu = actuation.nu
runningCostModel = crocoddyl.CostModelSum(state, nu)
terminalCostModel = crocoddyl.CostModelSum(state, nu)

# Costs
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([0.1] * 3 + [1000.] * 3 + [1000.] * robot_model.nv))
uResidual = crocoddyl.ResidualModelControl(state, nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
goalTrackingResidual = crocoddyl.ResidualModelFramePlacement(state, robot_model.getFrameId("base_link"),
                                                             pinocchio.SE3(target_quat.matrix(), target_pos), nu)
goalTrackingCost = crocoddyl.CostModelResidual(state, goalTrackingResidual)
runningCostModel.addCost("xReg", xRegCost, 1e-6)
runningCostModel.addCost("uReg", uRegCost, 1e-6)
runningCostModel.addCost("trackPose", goalTrackingCost, 1e-2)
terminalCostModel.addCost("goalPose", goalTrackingCost, 3.)

dt = 3e-2
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel), dt)

# Creating the shooting problem and the FDDP solver
T = 33
x0 = np.concatenate([hector.q0, np.zeros(state.nv)])
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

usol = np.load("quadror_us.npy")


N_iter = 100


ddp_reccord = []

ddp = crocoddyl.SolverFDDP(problem)
ddp.th_stop = 1e-20
ddp.setCallbacks([crocoddyl.CallbackVerbose()])

xs = [x0] * (T+1)
us = [np.zeros(nu)] * T 

for i in range(N_iter):
    xs, us = ddp.xs, ddp.us
    ddp_reccord.append(np.linalg.norm(np.array(us) - usol))
    ddp.solve(xs, us, isFeasible=False, maxiter=1)


GNMS_reccord = []

GNMS = GNMSCPP(problem)
GNMS.th_stop = 1e-20
GNMS.setCallbacks([crocoddyl.CallbackVerbose()])

xs = [x0] * (T+1)
us = [np.zeros(nu)] * T 

for i in range(N_iter):
    xs, us = GNMS.xs, GNMS.us
    GNMS_reccord.append(np.linalg.norm(np.array(us) - usol))

    GNMS.solve(xs, us, maxiter=1)

usol = np.save("quadror_us.npy", np.array(ddp.us))

import matplotlib.pyplot as plt

plt.plot(np.array(GNMS_reccord), label="gnms")
plt.plot(np.array(ddp_reccord), label="ddp")
plt.yscale("log")
plt.legend()
plt.show()