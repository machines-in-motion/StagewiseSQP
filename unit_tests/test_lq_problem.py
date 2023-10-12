import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'python'
os.sys.path.insert(1, str(python_path))

import numpy as np
import crocoddyl
import matplotlib.pyplot as plt

from sqp_ocp.solvers import CSSQP, SSQP
from sqp_ocp.constraint_model import StateConstraintModel, NoConstraintModel

LINE_WIDTH = 100


class PointMassDynamics:
    def __init__(self):
        self.g = np.array([0.0, -9.81])
        self.mass = 1
        self.nq = 2
        self.nv = 2
        self.ndx = 2
        self.nx = self.nq + self.nv
        self.nu = 2
        self.c_drag = 0.0

    def nonlinear_dynamics(self, x, u):
        return (1 / self.mass) * u + self.g - self.c_drag * x[2:] ** 2

    def derivatives(self, x, u):
        """returns df/dx evaluated at x,u"""
        dfdx = np.zeros([self.nv, self.ndx])
        dfdu = np.zeros([self.nv, self.nu])
        dfdu[0, 0] = 1.0 / self.mass
        dfdu[1, 1] = 1.0 / self.mass
        dfdx[0, 2] = -2.0 * self.c_drag * x[2]
        dfdx[1, 3] = -2.0 * self.c_drag * x[3]
        return dfdx, dfdu


class DifferentialActionModelLQ(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, dt=1e-2, isTerminal=False):
        self.dynamics = PointMassDynamics()
        state = crocoddyl.StateVector(self.dynamics.nx)
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, state, self.dynamics.nu, self.dynamics.ndx
        )
        self.ndx = self.state.ndx
        self.isTerminal = isTerminal
        self.mass = self.dynamics.mass
        self.dt = dt

    def _running_cost(self, x, u):
        cost = (x[0] - 1.0) ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2
        cost += u[0] ** 2 + u[1] ** 2
        return cost

    def _terminal_cost(self, x, u):
        cost = (
            200 * ((x[0] - 1.0) ** 2)
            + 200 * (x[1] ** 2)
            + 10 * (x[2] ** 2)
            + 10 * (x[3] ** 2)
        )
        return cost

    def calc(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)

        if self.isTerminal:
            data.cost = self._terminal_cost(x, u)
            data.xout = np.zeros(self.state.nv)
        else:
            data.cost = self._running_cost(x, u)
            data.xout = self.dynamics.nonlinear_dynamics(x, u)

    def calcDiff(self, data, x, u=None):
        Fx = np.zeros([2, 4])
        Fu = np.zeros([2, 2])

        Lx = np.zeros([4])
        Lu = np.zeros([2])
        Lxx = np.zeros([4, 4])
        Luu = np.zeros([2, 2])
        Lxu = np.zeros([4, 2])
        if self.isTerminal:
            Lx[0] = 400.0 * (x[0] - 1)
            Lx[1] = 400.0 * x[1]
            Lx[2] = 20.0 * x[2]
            Lx[3] = 20.0 * x[3]
            Lxx[0, 0] = 400.0
            Lxx[1, 1] = 400.0
            Lxx[2, 2] = 20.0
            Lxx[3, 3] = 20.0
        else:
            Lx[0] = 2.0 * (x[0] - 1)
            Lx[1] = 2.0 * x[1]
            Lx[2] = 2.0 * x[2]
            Lx[3] = 2.0 * x[3]
            Lu[0] = 2 * u[0]
            Lu[1] = 2 * u[1]
            Lxx[0, 0] = 2
            Lxx[1, 1] = 2
            Lxx[2, 2] = 2
            Lxx[3, 3] = 2
            Luu[0, 0] = 2.0
            Luu[1, 1] = 2

            Fu[0, 0] = 1.0 / self.mass
            Fu[1, 1] = 1.0 / self.mass
            Fx[0, 2] = -2.0 * self.dynamics.c_drag * x[2]
            Fx[1, 3] = -2.0 * self.dynamics.c_drag * x[3]

        data.Fx = Fx.copy()
        data.Fu = Fu.copy()
        data.Lx = Lx.copy()
        data.Lu = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = Lxu.copy()



if __name__ == "__main__":
    lq_diff_running = DifferentialActionModelLQ()
    lq_diff_terminal = DifferentialActionModelLQ(isTerminal=True)
    dt = 0.1
    horizon = 100
    x0 = np.zeros(4)
    lq_running = crocoddyl.IntegratedActionModelEuler(lq_diff_running, dt)
    lq_terminal = crocoddyl.IntegratedActionModelEuler(lq_diff_terminal, dt)

    problem = crocoddyl.ShootingProblem(x0, [lq_running] * horizon, lq_terminal)


    nx = 4
    nu = 2
    
    
    lxmin = -np.inf*np.ones(nx)
    lxmax = np.array([0.5, 0.1, np.inf, np.inf])
    ConstraintModel = StateConstraintModel(lq_diff_running.state, nu, lxmin, lxmax, "state")
    xs = [10*np.ones(4)] * (horizon + 1)
    us = [np.ones(2)*100 for t in range(horizon)] 

    print("TEST 1: SQP = CSSQP with sigma = 0".center(LINE_WIDTH, "-"))

    ddp1 = SSQP(problem)
    converged = ddp1.solve(xs, us, 1)

    ddp2 = CSSQP(problem, [NoConstraintModel(lq_diff_running.state, nu, "None")]*(horizon+1), "StagewiseQP", verbose = False)
    ddp2.sigma_sparse = 0.0
    converged = ddp2.solve(xs, us, 1)

    assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp2.xs)) < 1e-8, "Test failed"
    assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp2.us)) < 1e-8, "Test failed"


    print("TEST SQP 1 iter : CSSQP = StagewiseQPKKT".center(LINE_WIDTH, "-"))

    ddp1 = CSSQP(problem, [ConstraintModel]*(horizon+1), "StagewiseQP", verbose = False)

    ddp1.verbose = True
    converged = ddp1.solve(xs, us, 10)

    ddp2 = CSSQP(problem, [ConstraintModel]*(horizon+1), "StagewiseQPKKT", verbose = False)
    converged = ddp2.solve(xs, us, 1)
    

    assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp2.xs)) < 1e-8, "Test failed"
    assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp2.us)) < 1e-8, "Test failed"

    print("ALL TEST PASSED".center(LINE_WIDTH, "-"))
    print("\n")
