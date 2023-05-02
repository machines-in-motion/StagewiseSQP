import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'python'
os.sys.path.insert(1, str(python_path))

import numpy as np
import crocoddyl
import matplotlib.pyplot as plt
from sqp_ocp.solvers import FADMM
from sqp_ocp.constraint_model import StateConstraintModel, NoConstraint
from lq_problem import DifferentialActionModelLQ

LINE_WIDTH = 100


dt = 0.05
horizon = 2
x0 = np.zeros(4)

# lq_running = crocoddyl.ActionModelLQR(4, 2, 1)
# lq_terminal = crocoddyl.ActionModelLQR(4, 2, 1)

lq_diff_running = crocoddyl.DifferentialActionModelLQR(2, 2, 1)
lq_diff_terminal = crocoddyl.DifferentialActionModelLQR(2, 2, 1)

lq_running = crocoddyl.IntegratedActionModelEuler(lq_diff_running, dt)
lq_terminal = crocoddyl.IntegratedActionModelEuler(lq_diff_terminal, dt)

# lq_running = crocoddyl.IntegratedActionModelRK(lq_diff_running, crocoddyl.RKType.four, dt)
# lq_terminal = crocoddyl.IntegratedActionModelRK(lq_diff_terminal, crocoddyl.RKType.four, dt) 
problem = crocoddyl.ShootingProblem(x0, [lq_running] * horizon, lq_terminal)

nx = 4
nu = 2

lxmin = -np.inf*np.ones(nx)
lxmax = np.array([0.5, 0.1, np.inf, np.inf])
ConstraintModel = [NoConstraint(4, 2)] + [StateConstraintModel(lxmin, lxmax, 4, 4, 2)] * horizon
xs = [10*np.ones(4)] * (horizon + 1)
us = [np.ones(2)*100 for t in range(horizon)] 


ddpc = crocoddyl.SolverFADMM(problem,ConstraintModel)
xs[0] = x0
ddpc.setCandidate(xs, us, False)
# print(np.array(ddp.xs))

ddpc.max_qp_iters = 1
ddpc.calc(True)
# for i in range(2):
ddpc.backwardPass()
# ddpc.forwardPass()

#     ddp.update_lagrangian_parameters()
#     ddp.update_rho_sparse(0)
# ddpc.computeDirection(True)

# ddp.solve(xs, us, 1)
# print("PYTHON")



ddp = FADMM(problem, ConstraintModel, True)
ddp.setCandidate(xs, us, False)
# ddp.solve(xs, us, 1)

ddp.calc(True)
# # for i in range(2):
ddp.backwardPass()
# ddpc.forwardPass()

# print(np.array(ddpc.dx_tilde), ddp.dx_tilde)

# if True:
#     plt.figure("trajectory plot")
#     plt.plot(np.array(ddp.xs)[:, 0], np.array(ddp.xs)[:, 1], label="FADMM")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.title("DDP")
#     plt.legend()
#     plt.show()



