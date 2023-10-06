import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'python'
os.sys.path.insert(1, str(python_path))

import numpy as np
import crocoddyl
import matplotlib.pyplot as plt
from sqp_ocp.solvers import CSSQP, QPSolvers
from sqp_ocp.constraint_model import StateConstraintModel, NoConstraintModel
from test_lq_problem import DifferentialActionModelLQ

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
ConstraintModel = [NoConstraintModel(4, 2)] + [StateConstraintModel(lxmin, lxmax, 4, 4, 2)] * horizon
xs = [10*np.ones(4)] * (horizon + 1)
us = [np.ones(2)*100 for t in range(horizon)] 


print("TEST LQ PROBLEM : CSSQP = StagewiseQPKKT".center(LINE_WIDTH, "-"))

ddp1 = CSSQP(problem, ConstraintModel, verbose = False)

ddp2 = QPSolvers(problem, ConstraintModel, "StagewiseQPKKT", verbose = False)


converged = ddp1.solve(xs, us, 1)
converged = ddp2.solve(xs, us, 1)


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



