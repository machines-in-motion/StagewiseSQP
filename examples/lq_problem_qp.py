import pathlib
import os
python_path = pathlib.Path('.').absolute().parent.parent/'python'
os.sys.path.insert(1, str(python_path))

import numpy as np
import crocoddyl
import matplotlib.pyplot as plt
from sqp_ocp.solvers import FADMM
from sqp_ocp.constraint_model import StateConstraintModel, NoConstraint
from lq_problem import DifferentialActionModelLQ

LINE_WIDTH = 100

lq_diff_running = DifferentialActionModelLQ()
lq_diff_terminal = DifferentialActionModelLQ(isTerminal=True)
dt = 0.05
horizon = 100
x0 = np.zeros(4)
lq_running = crocoddyl.IntegratedActionModelEuler(lq_diff_running, dt)
lq_terminal = crocoddyl.IntegratedActionModelEuler(lq_diff_terminal, dt)
problem = crocoddyl.ShootingProblem(x0, [lq_running] * horizon, lq_terminal)

nx = 4
nu = 2

lxmin = -np.inf*np.ones(nx)
lxmax = np.array([0.5, 0.1, np.inf, np.inf])
ConstraintModel = [NoConstraint(4, 2)] + [StateConstraintModel(lxmin, lxmax, 4, 4, 2)] * horizon

print("\n\n")
print("TEST : FADMM = FAdmmKKT".center(LINE_WIDTH, "-"))

ddp = FADMM(problem, ConstraintModel)
xs = [10*np.ones(4)] * (horizon + 1)
us = [np.ones(2)*100 for t in range(horizon)] 

converged = ddp.solve(xs, us, 1)

if True:
    plt.figure("trajectory plot")
    plt.plot(np.array(ddp.xs)[:, 0], np.array(ddp.xs)[:, 1], label="FADMM")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("DDP")
    plt.legend()
    plt.show()



