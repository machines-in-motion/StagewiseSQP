import numpy as np
import crocoddyl
import matplotlib.pyplot as plt
from SQP import SQP
from SQP_cpp import CSSQPCPP

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
        self.c_drag = 0. 

    def nonlinear_dynamics(self, x, u):
        return (1 / self.mass) * u + self.g - self.c_drag * x[2:] ** 2

    def derivatives(self, x, u):
        """returns df/dx evaluated at x,u"""
        dfdx = np.zeros([self.nv, self.ndx])
        dfdu = np.zeros([self.nv, self.nu])
        dfdu[0, 0] = 1.0 / self.mass
        dfdu[1, 1] = 1.0 / self.mass
        return dfdx, dfdu


class DifferentialActionModelCliff(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, dt=1e-2, isTerminal=False):
        self.dynamics = PointMassDynamics()
        state = crocoddyl.StateVector(self.dynamics.nx)
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, state, self.dynamics.nu, self.dynamics.ndx
        )
        self.ndx = self.state.ndx
        self.isTerminal = isTerminal
        self.mass = 1.0
        self.dt = dt

    def _running_cost(self, x, u):
        cost = 0.1 / ((0.1 * x[1] + 1.0) ** 10) + 1e-3 * u[0] ** 2 + 1e-5 * u[1] ** 2
        return cost / self.dt

    def _terminal_cost(self, x, u):
        cost = 20 * ((x[0] - 10.0) ** 2 + x[1] ** 2) + x[2] ** 2 + x[3] ** 2
        return cost

    def calc(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)

        if self.isTerminal:
            data.cost = self._terminal_cost(x, u)
            data.xout = np.zeros(2)
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
            Lx[0] = 40. * (x[0] - 10)
            Lx[1] = 40. * x[1]
            Lx[2] = 2. * x[2]
            Lx[3] = 2. * x[3]
            Lxx[0, 0] = 40.
            Lxx[1, 1] = 40.
            Lxx[2, 2] = 2.
            Lxx[3, 3] = 2.
        else:
            Lx[1] = -0.1 / (1 + 0.1 * x[1]) ** 11 / self.dt
            Lu[0] = 2e-3 * u[0] / self.dt
            Lu[1] = 2e-5 * u[1] / self.dt
            Lxx[1, 1] = 0.11 / (0.1 * x[1] + 1.0) ** 12 / self.dt
            Luu[0, 0] = 2e-3 / self.dt
            Luu[1, 1] = 2e-5 / self.dt
            #
            Fu[0, 0] = 1.0 / self.mass
            Fu[1, 1] = 1.0 / self.mass
            #
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
    print(" Testing Point Mass Cliff with DDP ".center(LINE_WIDTH, "#"))
    cliff_diff_running = DifferentialActionModelCliff()
    cliff_diff_terminal = DifferentialActionModelCliff(isTerminal=True)
    print(" Constructing differential models completed ".center(LINE_WIDTH, "-"))
    dt = 0.01
    T = 100
    x0 = np.zeros(4)
    cliff_running = crocoddyl.IntegratedActionModelEuler(cliff_diff_running, dt)
    cliff_terminal = crocoddyl.IntegratedActionModelEuler(cliff_diff_terminal, dt)
    print(" Constructing integrated models completed ".center(LINE_WIDTH, "-"))

    problem = crocoddyl.ShootingProblem(x0, [cliff_running] * T, cliff_terminal)
    print(" Constructing shooting problem completed ".center(LINE_WIDTH, "-"))

    # ddp = SQP(problem)
    ddp = CSSQPCPP(problem)
    print(" Constructing DDP solver completed ".center(LINE_WIDTH, "-"))
    # ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
    xs = [x0] * (T + 1)
    us = [np.zeros(2)] * T
    converged = ddp.solve(xs, us)
    if converged:
        print(" DDP solver has CONVERGED ".center(LINE_WIDTH, "-"))
        plt.figure("trajectory plot")
        plt.plot(np.array(ddp.xs)[:, 0], np.array(ddp.xs)[:, 1], label="ddp")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("DDP")
        plt.show()