
import pathlib
import os
import sys
python_path = pathlib.Path('.').absolute().parent.parent/'python'
os.sys.path.insert(1, str(python_path))

import numpy as np
import crocoddyl
import matplotlib.pyplot as plt
from sqp_ocp.solvers import GNMSCPP,GNMS

LINE_WIDTH = 100


class QuadrotorDynamics:
    def __init__(self):
        self.g = 9.81
        self.nq = 3
        self.nv = 3
        self.ndx = 3
        self.nx = self.nq + self.nv
        self.nu = 2

    def nonlinear_dynamics(self, x, u):
        x_ddot = -(u[0] + u[1]) * np.sin(x[2])
        y_ddot = (u[0] + u[1]) * np.cos(x[2]) - self.g
        th_ddot = 0.1 * (u[0] - u[1])
        return np.array([x_ddot, y_ddot, th_ddot])


class DifferentialActionModelQuadrotor(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, dt=5e-2, isTerminal=False):
        self.dynamics = QuadrotorDynamics()
        state = crocoddyl.StateVector(self.dynamics.nx)
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, self.dynamics.nu, self.dynamics.ndx)
        self.ndx = self.state.ndx
        self.isTerminal = isTerminal
        self.mass = 1.0
        self.g = self.dynamics.g
        self.dt = dt

    def barrier_cost(self, x):
        cost = 6 * np.exp(- 10 * (x[0] - 1) ** 2 - 0.5 * (x[1] + 0.1) ** 2 )
        return cost

    def _running_cost(self, x, u):
        cost = 1e0 * (x[0] - 2) ** 2 + 1e0 * x[1] ** 2 + 1e0 * x[2] ** 2
        cost += 1e-2 * x[3] ** 2 + 1e-2 * x[4] ** 2 + 1e-2 * x[5] ** 2
        cost += 1e-1 * (u[0] - self.g / 2) ** 2 + 1e-1 * (u[1] - self.g / 2) ** 2
        return cost + self.barrier_cost(x)

    def _terminal_cost(self, x, u):
        cost = 1e2 * (x[0] - 2) ** 2 + 1e2 * x[1] ** 2 + 1e2 * x[2] ** 2
        cost += 1e0 * x[3] ** 2 + 1e0 * x[4] ** 2 + 1e0 * x[5] ** 2
        return cost + self.barrier_cost(x)

    def calc(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)

        if self.isTerminal:
            data.cost = self._terminal_cost(x, u)
            data.xout = np.zeros(3)
        else:
            data.cost = self._running_cost(x, u)
            data.xout = self.dynamics.nonlinear_dynamics(x, u)

    def calcDiff(self, data, x, u=None):
        Fx = np.zeros([3, 6])
        Fu = np.zeros([3, 2])

        Lx = np.zeros([6])
        Lu = np.zeros([2])
        Lxx = np.zeros([6, 6])
        Luu = np.zeros([2, 2])
        Lxu = np.zeros([6, 2])
        if self.isTerminal:
            Lx[0] = 2e2 * (x[0] - 2)
            Lx[1] = 2e2 * x[1]
            Lx[2] = 2e2 * x[2]
            Lxx[0, 0] = 2e2
            Lxx[1, 1] = 2e2
            Lxx[2, 2] = 2e2
            Lx[3] = 2e0 * x[3]
            Lx[4] = 2e0 * x[4]
            Lx[5] = 2e0 * x[5]
            Lxx[3, 3] = 2e0
            Lxx[4, 4] = 2e0
            Lxx[5, 5] = 2e0
        else:
            Lx[0] = 2e0 * (x[0] - 2)
            Lx[1] = 2e0 * x[1]
            Lx[2] = 2e0 * x[2]
            Lxx[0, 0] = 2e0
            Lxx[1, 1] = 2e0
            Lxx[2, 2] = 2e0
            Lx[3] = 2e-2 * x[3]
            Lx[4] = 2e-2 * x[4]
            Lx[5] = 2e-2 * x[5]
            Lxx[3, 3] = 2e-2
            Lxx[4, 4] = 2e-2
            Lxx[5, 5] = 2e-2
            #
            Lu[0] = 2e-1 * (u[0] - self.g / 2)
            Lu[1] = 2e-1 * (u[1] - self.g / 2)
            Luu[0, 0] = 2e-1
            Luu[1, 1] = 2e-1
            #
            Fu[0, 0] = -np.sin(x[2])
            Fu[0, 1] = -np.sin(x[2])
            #
            Fu[1, 0] = np.cos(x[2])
            Fu[1, 1] = np.cos(x[2])
            #
            Fu[2, 0] = 0.1
            Fu[2, 1] = -0.1
            #
            Fx[0, 2] = -(u[0] + u[1]) * np.cos(x[2])
            Fx[1, 2] = -(u[0] + u[1]) * np.sin(x[2])


        Lx[0] += -2 * (x[0] - 1) * self.barrier_cost(x) * 10
        Lx[1] += -2 * (x[1] + 0.1) * self.barrier_cost(x) * 0.5

        Lxx[0, 0] += - 2 * self.barrier_cost(x) * 10 + 4 * (x[0] - 1) ** 2 * self.barrier_cost(x) * 100
        Lxx[1, 1] += - 2 * self.barrier_cost(x) * 0.5      + 4 * (x[1] + 0.1) ** 2 * self.barrier_cost(x) * 0.5  * 0.5

        Lxx[0, 1] += 4 * (x[0] - 1) * (x[1] + 0.1) * self.barrier_cost(x) * 10 * 0.5
        Lxx[1, 0] += 4 * (x[0] - 1) * (x[1] + 0.1) * self.barrier_cost(x) * 10 * 0.5

        data.Fx = Fx.copy()
        data.Fu = Fu.copy()
        data.Lx = Lx.copy()
        data.Lu = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = Lxu.copy()


class IntegratedActionModelQuadrotor(crocoddyl.IntegratedActionModelRK): 
    def __init__(self, diffModel, dt=1.e-2):
        super().__init__(diffModel, crocoddyl.RKType.four, dt)
        self.diffModel = diffModel 
        self.intModel = crocoddyl.IntegratedActionModelRK(self.diffModel, crocoddyl.RKType.four, stepTime=dt) 

        self.Fxx = np.zeros([self.state.ndx, self.state.ndx, self.state.ndx])
        self.Fxu = np.zeros([self.state.ndx, self.state.ndx, self.nu])
        self.Fuu = np.zeros([self.state.ndx, self.nu, self.nu])
    
    def calcFxx(self, x, u): 
        # Euler integration
        self.Fxx[3, 2, 2] = (u[0] + u[1]) * np.sin(x[2]) * self.dt
        self.Fxx[4, 2, 2] = - (u[0] + u[1]) * np.cos(x[2]) * self.dt
    
    def calcFxu(self, x, u): 
        # Euler integration
        self.Fxu[3, 2, 0] = - np.cos(x[2]) * self.dt
        self.Fxu[3, 2, 1] = - np.cos(x[2]) * self.dt
        #
        self.Fxu[4, 2, 0] = - np.sin(x[2]) * self.dt
        self.Fxu[4, 2, 1] = - np.sin(x[2]) * self.dt


    def calc(self, data, x, u=None):
        if u is None:
            self.intModel.calc(data, x)
        else:
            self.intModel.calc(data, x, u)
        
    def calcDiff(self, data, x, u=None):
        if u is None:
            self.intModel.calcDiff(data, x)
            u = np.zeros(self.nu)
        else:
            self.intModel.calcDiff(data, x, u)
        self.calcFxx(x,u)
        self.calcFxu(x,u)
        

if __name__ == "__main__":
    print(" Testing Quadrotor with DDP ".center(LINE_WIDTH, "#"))
    quadrotor_diff_running = DifferentialActionModelQuadrotor()
    quadrotor_diff_terminal = DifferentialActionModelQuadrotor(isTerminal=True)
    print(" Constructing differential models completed ".center(LINE_WIDTH, "-"))
    dt = 0.05
    T = 60
    x0 = np.array([0, 0, 0, 0, 0, 0])
    quadrotor_running = IntegratedActionModelQuadrotor(quadrotor_diff_running, dt)
    quadrotor_terminal = IntegratedActionModelQuadrotor(quadrotor_diff_terminal, dt)
    print(" Constructing integrated models completed ".center(LINE_WIDTH, "-"))

    problem = crocoddyl.ShootingProblem(x0, [quadrotor_running] * T, quadrotor_terminal)
    print(" Constructing shooting problem completed ".center(LINE_WIDTH, "-"))

    # ddp = crocoddyl.SolverDDP(problem)
    ddp = GNMS(problem)

    print(" Constructing DDP solver completed ".center(LINE_WIDTH, "-"))
    ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
    xs = [x0] * (T + 1)
    us = [10*np.ones(2)] * T
    converged = ddp.solve(xs, us, maxiter=20)
    print(converged)

    if converged:
        print(" DDP solver has CONVERGED ".center(LINE_WIDTH, "-"))
    else:
        print(" DDP solver DID NOT CONVERGED ".center(LINE_WIDTH, "-"))

    plt.figure()
    plt.plot(np.array(ddp.xs)[:, 0], np.array(ddp.xs)[:, 1], label="ddp")
    plt.xlabel(r"$p_x$ [m]")
    plt.ylabel(r"$p_y$ [m]")
    plt.legend()
    plt.grid()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(np.array(ddp.xs)[:, 0], label="x")
    ax2.plot(np.array(ddp.xs)[:, 1], label="y")
    ax3.plot(np.array(ddp.xs)[:, 2], label="theta")
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.set_ylabel(r"$p_x$")
    ax2.set_ylabel(r"$p_y$")
    ax3.set_ylabel(r"$\theta$")

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(np.array(ddp.us)[:, 0], label="F1")
    ax2.plot(np.array(ddp.us)[:, 1], label="F2")
    ax1.grid()
    ax2.grid()
    ax1.set_ylabel(r"$u_1$")
    ax2.set_ylabel(r"$u_2$")
    plt.show()