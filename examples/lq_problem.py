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


class PointMassDynamics():
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
    
    xs = [10*np.ones(4)] * (horizon + 1)
    us = [np.ones(2)*100 for t in range(horizon)] 


    ddp1 = SSQP(problem)
    converged = ddp1.solve(xs, us, 1)

    
    lxmin = -np.inf*np.ones(nx)
    xlim = 0.4
    ylim = 0.2
    lxmax = np.array([xlim, ylim, np.inf, np.inf])
    ConstraintModel = StateConstraintModel(lq_diff_running.state, lq_diff_running.nu, lxmin, lxmax, 'stateConstraint')
    ConstraintModels = [NoConstraintModel(lq_diff_running.state, lq_diff_running.nu, "none")] + [ConstraintModel] * horizon

    ddp2 = CSSQP(problem, ConstraintModels, "StagewiseQP")
    converged = ddp2.solve(xs, us, 1)

    plt.figure("trajectory plot")
    plt.plot(np.array(ddp1.xs)[:, 0], np.array(ddp1.xs)[:, 1], label="LQP")
    plt.plot(np.array(ddp2.xs)[:, 0], np.array(ddp2.xs)[:, 1], label="Constrained LQR")
    plt.plot([xlim] * (horizon+1), np.array(ddp2.xs)[:, 1], "--", color="black", label="x bound")
    plt.plot(np.array(ddp2.xs)[:, 0], [ylim] * (horizon+1), "--", color="black", label="y bound")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
