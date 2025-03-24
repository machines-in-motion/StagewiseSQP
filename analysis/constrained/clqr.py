import numpy as np
import crocoddyl
import mim_solvers
import matplotlib.pyplot as plt


class ActionModelCLQR(crocoddyl.ActionModelAbstract):
    def __init__(self, nx, isInitial=False, isTerminal=False):
        nr = 0 
        ng = int(nx/2)
        nu = int(nx/2)
        nh = 0
        self.isTerminal = isTerminal
        self.isInitial = isInitial
        self.nx = nx
        

        if self.isInitial:
            assert not isTerminal

        state = crocoddyl.StateVector(self.nx)
        crocoddyl.ActionModelAbstract.__init__(self, state, nu, nr, ng, nh)

        if not self.isInitial:
            lower_bound = np.array([-10.] * ng)
            upper_bound = np.array([10.]* ng)

            self.g_lb = lower_bound
            self.g_ub = upper_bound

        Lx = np.random.random((self.nx, self.nx))
        self.Q = Lx.T @ Lx + np.eye(self.nx)


        Lu = np.random.random((self.nu, self.nu))
        self.R = Lu.T @ Lu + np.eye(self.nu)
        
        self.Lxu = np.random.random((self.nx, self.nu))

        self.x_star = np.random.random(self.nx)
        self.u_star = np.random.random(self.nu)



        self.Fx = np.random.random((self.nx, self.nx))
        self.Fu = np.random.random((self.nx, self.nu))
        self.Cx = np.random.random((ng, self.nx)) - 0.5
        self.Cu = np.random.random((ng, self.nu)) - 0.5

    def _running_cost(self, x, u):
        return 0.5 * ((x - self.x_star).T @ self.Q @ (x - self.x_star) + (u - self.u_star).T @ self.R @ (u - self.u_star) + (x - self.x_star).T @ self.Lxu @ (u - self.u_star))

    def _terminal_cost(self, x):
        return 0.5 * (x - self.x_star).T @ self.Q @ (x - self.x_star) 

    def calc(self, data, x, u=None):

        if self.isTerminal:
            data.cost = self._terminal_cost(x)
            data.xnext = np.zeros(self.state.nx)
        else:
            data.cost = self._running_cost(x, u)
            data.xnext = self.Fx @ x + self.Fu @ u

        if self.isTerminal:
            data.g = self.Cx @ x
        elif self.isInitial:
            data.g = self.Cu @ u
        else:
            data.g = self.Cx @ x + self.Cu @ u

    def calcDiff(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)
            
        Fx = self.Fx
        Fu = self.Fu

        
        Lx = self.Q @ (x - self.x_star) + 0.5 * self.Lxu @ (u - self.u_star)
        Lu = self.R @ (u - self.u_star) + 0.5 * self.Lxu.T @ (x - self.x_star)
        Lxx = self.Q
        Luu = self.R
        Lxu = self.Lxu

        data.Fx  = Fx.copy()
        data.Fu  = Fu.copy()
        data.Lx  = Lx.copy()
        data.Lu  = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = 0.5 * Lxu.copy()

        if self.isTerminal:
            data.Gx = self.Cx
        if self.isInitial:
            data.Gu = self.Cu   
        else:
            data.Gx = self.Cx
            data.Gu = self.Cu   


if __name__ == "__main__":

    nx = 4
    clqr_initial  = ActionModelCLQR(nx, isInitial=True)
    clqr_running  = ActionModelCLQR(nx, )
    clqr_terminal = ActionModelCLQR(nx, isTerminal=True)

    horizon = 100
    x0 = np.zeros(4)
    problem = crocoddyl.ShootingProblem(
        x0, [clqr_initial] + [clqr_running] * (horizon - 1), clqr_terminal
    )


    xs = [x0] + [10 * np.ones(4)] * (horizon)
    us = [np.ones(2) * 100 for t in range(horizon)]

    # Constrained solver
    csolver = mim_solvers.SolverCSQP(problem)
    csolver.termination_tolerance = 1e-4
    csolver.with_callbacks = True
    # csolver.with_qp_callbacks = True

    csolver.max_qp_iters = 10000
    csolver.eps_abs = 1e-10
    csolver.eps_rel = 0.0

    # Perform one iteration and check KKT criteria
    csolver.solve(xs, us, 1)
    csolver.calc(True)
    csolver.checkKKTConditions()
    assert csolver.KKT < 1e-4
    