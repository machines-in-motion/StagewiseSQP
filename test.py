import numpy as np
import crocoddyl
import matplotlib.pyplot as plt
from clqr import CLQR
from cilqr import CILQR
from gnms_cpp import GNMSCPP
from constraintmodel import StateConstraintModel, NoConstraint
from gnms import GNMS
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


# print("TEST 1: GNMS = SparceADMM with sigma = 0".center(LINE_WIDTH, "-"))

# ddp1 = GNMS(problem)
# ddp2 = CILQR(problem, [NoConstraint()]*(horizon+1), "sparceADMM")
# CILQR.sigma_sparse = 0.0

# xs = [10*np.ones(4)] * (horizon + 1)
# us = [np.ones(2)*100 for t in range(horizon)] 

# converged = ddp1.solve(xs, us, 1)
# converged = ddp2.solve(xs, us, 1)

# assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp2.xs)) < 1e-8, "Test failed"
# assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp2.us)) < 1e-8, "Test failed"



print("\n\n")
print("TEST 2: BOYD = SparceADMM".center(LINE_WIDTH, "-"))

ddp1 = CLQR(problem, ConstraintModel, "sparceADMM")
ddp2 = CLQR(problem, ConstraintModel, "Boyd")
xs = [10*np.ones(4)] * (horizon + 1)
us = [np.ones(2)*100 for t in range(horizon)] 

converged = ddp1.solve(xs, us, 1)
converged = ddp2.solve(xs, us, 1)



##### UNIT TEST #####################################

for i in range(2):
  print(np.array(ddp1.xs)[i], np.array(ddp2.xs)[i])

set_tol = 1e-6
assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp2.xs)) < set_tol, "Test failed"

assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp2.us)) < set_tol, "Test failed"


d_relaxed = np.array(ddp1.dz_relaxed[1:]).flatten()
assert np.linalg.norm(d_relaxed - np.array(ddp2.x_k_1)) < set_tol, "Test failed"

z = np.array(ddp1.z[1:]).flatten()
assert np.linalg.norm(z - np.array(ddp2.z_k)) < set_tol, "Test failed"

y = np.array(ddp1.y[1:]).flatten()
assert np.linalg.norm(y - np.array(ddp2.y_k)) < set_tol, "Test failed"

rho = np.array(ddp1.rho_vec[1:]).flatten()

assert np.linalg.norm(rho - np.array(ddp2.rho_vec_boyd)) < set_tol, "Test failed"

assert np.linalg.norm(ddp1.rho_estimate_sparse - ddp2.rho_estimate_boyd) < set_tol, "Test failed"


print("\n\n\n\n ALL TESTS PASSED")

if True:
    plt.figure("trajectory plot")
    plt.plot(np.array(ddp1.xs)[:, 0], np.array(ddp1.xs)[:, 1], label="sparceADMM")
    plt.plot(np.array(ddp2.xs)[:, 0], np.array(ddp2.xs)[:, 1], label="Boyd")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("DDP")
    plt.legend()
    plt.show()



