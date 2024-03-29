import numpy as np
import crocoddyl
import mim_solvers
import matplotlib.pyplot as plt


class DiffActionModelCLQR(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, isInitial=False, isTerminal=False):
        self.nq = 2
        self.nv = 2
        self.ndx = 2
        self.nx = self.nq + self.nv
        nu = 2
        nr = 1
        ng = 4
        nh = 0
        self.isTerminal = isTerminal
        self.isInitial = isInitial

        if self.isInitial:
            ng = 0

        state = crocoddyl.StateVector(self.nx)
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, nu, nr, ng, nh)

        if not self.isInitial:
            lower_bound = np.array([-np.inf] * ng)
            upper_bound = np.array([0.4, 0.2, np.inf, np.inf])

            self.g_lb = lower_bound
            self.g_ub = upper_bound

        self.g = np.array([0.0, -9.81])

        self.x_weights_terminal = [200, 200, 10, 10]

    def _running_cost(self, x, u):
        cost = (x[0] - 1.0) ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2
        cost += u[0] ** 2 + u[1] ** 2
        return 0.5 * cost

    def _terminal_cost(self, x, u):
        cost = (
            self.x_weights_terminal[0] * ((x[0] - 1.0) ** 2)
            + self.x_weights_terminal[1] * (x[1] ** 2)
            + self.x_weights_terminal[2] * (x[2] ** 2)
            + self.x_weights_terminal[3] * (x[3] ** 2)
        )
        return 0.5 * cost

    def calc(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)

        if self.isTerminal:
            data.cost = self._terminal_cost(x, u)
            data.xnext = np.zeros(self.state.nx)
        else:
            data.cost = self._running_cost(x, u)
            data.xout = u + self.g

        if not self.isInitial:
            data.g = x.copy()

    def calcDiff(self, data, x, u=None):
        Fx = np.zeros((2, 4))
        Fu = np.eye(2)

        Lx = np.zeros([4])
        Lu = np.zeros([2])
        Lxx = np.zeros([4, 4])
        Luu = np.zeros([2, 2])
        Lxu = np.zeros([4, 2])
        if self.isTerminal:
            Lx[0] = self.x_weights_terminal[0] * (x[0] - 1)
            Lx[1] = self.x_weights_terminal[1] * x[1]
            Lx[2] = self.x_weights_terminal[2] * x[2]
            Lx[3] = self.x_weights_terminal[3] * x[3]
            Lxx[0, 0] = self.x_weights_terminal[0]
            Lxx[1, 1] = self.x_weights_terminal[1]
            Lxx[2, 2] = self.x_weights_terminal[2]
            Lxx[3, 3] = self.x_weights_terminal[3]
        else:
            Lx[0] = x[0] - 1
            Lx[1] = x[1]
            Lx[2] = x[2]
            Lx[3] = x[3]
            Lu[0] = u[0]
            Lu[1] = u[1]
            Lxx = np.eye(4)
            Luu = np.eye(2)

        data.Fx = Fx.copy()
        data.Fu = Fu.copy()
        data.Lx = Lx.copy()
        data.Lu = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = Lxu.copy()

        if not self.isInitial:
            data.Gx = np.eye(self.nx)
            data.Gu = np.zeros((self.nx, self.nu))


if __name__ == "__main__":
    diff_clqr_initial = DiffActionModelCLQR(isInitial=True)
    diff_clqr_running = DiffActionModelCLQR()
    diff_clqr_terminal = DiffActionModelCLQR(isTerminal=True)

    dt = 0.1
    clqr_initial = crocoddyl.IntegratedActionModelEuler(diff_clqr_initial, dt)
    clqr_running = crocoddyl.IntegratedActionModelEuler(diff_clqr_running, dt)
    clqr_terminal = crocoddyl.IntegratedActionModelEuler(diff_clqr_terminal, 0.0)

    horizon = 100
    x0 = np.zeros(4)
    problem = crocoddyl.ShootingProblem(
        x0, [clqr_initial] + [clqr_running] * (horizon - 1), clqr_terminal
    )

    nx = 4
    nu = 2

    xs = [x0] + [10 * np.ones(4)] * (horizon)
    us = [np.ones(2) * 100 for t in range(horizon)]

    max_iter = 10
    # Constrained solver
    csolver = mim_solvers.SolverCSQP(problem)
    csolver.termination_tolerance = 1e-4
    csolver.with_callbacks = True
    csolver.use_filter_line_search = False
    csolver.filter_size = max_iter
    csolver.eps_abs = 1e-8
    csolver.eps_rel = 0.0

    csolver.solve(xs, us, max_iter)

    # Constrained solver
    solver = mim_solvers.SolverSQP(problem)
    solver.termination_tolerance = 1e-4
    solver.with_callbacks = True
    solver.solve(xs, us, max_iter)

    lxmax = clqr_running.g_ub
    plt.figure("trajectory plot")
    plt.plot(
        np.array(solver.xs)[:, 0], np.array(solver.xs)[:, 1], linewidth=2, label="LQR"
    )
    plt.plot(x0[0], x0[1], "x", label="x0")
    plt.plot(
        np.array(csolver.xs)[:, 0],
        np.array(csolver.xs)[:, 1],
        linewidth=2,
        label="Constrained LQR",
    )

    plt.plot(
        [lxmax[0]] * (horizon + 1),
        np.array(csolver.xs)[:, 1],
        "-.",
        color="black",
        label="x bound",
    )
    plt.plot(
        np.array(csolver.xs)[:, 0],
        [lxmax[1]] * (horizon + 1),
        "-.",
        color="black",
        label="y bound",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
