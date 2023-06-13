
# Display the solution
import pathlib
import os
import sys
python_path = pathlib.Path('.').absolute().parent.parent/'python'
os.sys.path.insert(1, str(python_path))


import numpy as np
from cartpole_utils import animateCartpole
import crocoddyl
import matplotlib.pyplot as plt

class DifferentialActionModelCartpoleNODIFF(crocoddyl.DifferentialActionModelAbstract):

    def __init__(self):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, crocoddyl.StateVector(4), 1, 6)  # nu = 1; nr = 6
        self.unone = np.zeros(self.nu)

        self.m1 = 1.
        self.m2 = .1
        self.l = .5
        self.g = 9.81
        self.costWeights = [1., 1., 0.1, 0.001, 0.001, 1.]  # sin, 1-cos, x, xdot, thdot, f

    def calc(self, data, x, u=None):
        if u is None: u = model.unone
        # Getting the state and control variables
        y, th, ydot, thdot = x[0], x[1], x[2], x[3]
        f = u[0]

        # Shortname for system parameters
        m1, m2, l, g = self.m1, self.m2, self.l, self.g
        s, c = np.sin(th), np.cos(th)

        # Defining the equation of motions
        m = m1 + m2
        mu = m1 + m2 * s**2
        xddot = (f + m2 * c * s * g - m2 * l * s * thdot**2) / mu
        thddot = (c * f / l + m * g * s / l - m2 * c * s * thdot**2) / mu
        data.xout = np.matrix([xddot, thddot]).T

        # Computing the cost residual and value
        data.r = np.matrix(self.costWeights * np.array([s, 1 - c, y, ydot, thdot, f])).T
        data.cost = .5 * sum(np.asarray(data.r)**2)

    def calcDiff(self, data, x, u=None):
        # Advance user might implement the derivatives
        pass


class DifferentialActionModelCartpole(crocoddyl.DifferentialActionModelAbstract):

    def __init__(self):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, crocoddyl.StateVector(4), 1, 6)  # nu = 1; nr = 6
        self.unone = np.zeros(self.nu)

        self.m1 = 1.
        self.m2 = .1
        self.l = .5
        self.g = 9.81
        self.costWeights = [1., 1., 0.1, 0.001, 0.001, 1.]  # sin, 1-cos, x, xdot, thdot, f

    def calc(self, data, x, u=None):
        if u is None: u = model.unone
        # Getting the state and control variables
        y, th, ydot, thdot = x[0], x[1], x[2], x[3]
        f = u[0]

        # Shortname for system parameters
        m1, m2, l, g = self.m1, self.m2, self.l, self.g
        s, c = np.sin(th), np.cos(th)

        # Defining the equation of motions
        m = m1 + m2
        mu = m1 + m2 * s**2
        xddot = (f + m2 * c * s * g - m2 * l * s * thdot**2) / mu
        thddot = (c * f / l + m * g * s / l - m2 * c * s * thdot**2) / mu
        data.xout = np.matrix([xddot, thddot]).T

        # Computing the cost residual and value
        data.r = np.matrix(self.costWeights * np.array([s, 1 - c, y, ydot, thdot, f])).T
        data.cost = .5 * sum(np.asarray(data.r)**2)

    def calcDiff(self, data, x, u=None):

        y, th, ydot, thdot = x[0], x[1], x[2], x[3]
        f = u[0]

        # Shortname for system parameters
        m1, m2, l, g = self.m1, self.m2, self.l, self.g
        s, c = np.sin(th), np.cos(th)

        # Defining the equation of motions
        m = m1 + m2
        mu = m1 + m2 * s**2


        # dynamics
        Fx = np.zeros((2, 4))
        Fu = np.zeros((2,))

        x_num = (f + m2 * c * s * g - m2 * l * s * thdot**2)
        dx_num_dtheta = - m2 * s ** 2 * g + m2 * c * c * g - m2 * l * c * thdot**2
        ddenom_dtheta = - 2 * m2 * c * s / mu ** 2

        th_num = c * f / l + m * g * s / l - m2 * c * s * thdot**2
        dth_num_dtheta = -s * f / l + m * g * c / l + m2 * s * s * thdot**2 - m2 * c * c * thdot**2

        Fx[0, 0] = 0
        Fx[0, 1] = x_num * ddenom_dtheta + dx_num_dtheta / mu
        Fx[0, 2] = 0
        Fx[0, 3] = - 2 * m2 * l * s * thdot / mu
        Fx[1, 0] = 0
        Fx[1, 1] = th_num * ddenom_dtheta + dth_num_dtheta / mu
        Fx[1, 2] = 0
        Fx[1, 3] = - 2 * m2 * c * s * thdot / mu

        Fu[0] = 1 / mu
        Fu[1] = c  / l / mu

        data.Fx = Fx
        data.Fu = Fu
        
        # cost
        Lx = np.zeros((4))
        Lu = np.zeros((1,))
        Lxx = np.zeros((4, 4))
        Luu = np.zeros((1,))
        Lxu = np.zeros((4,))

        Lx[0] = self.costWeights[2]**2 * y
        Lx[1] = self.costWeights[0]**2 * c * s + self.costWeights[1] * s * (1-c)
        Lx[2] = self.costWeights[3]**2 * ydot
        Lx[3] = self.costWeights[4]**2 * thdot


        Lu[0] = self.costWeights[5] * u




        Lx[0] = self.costWeights[2]**2 * y
        Lx[1] = self.costWeights[0]**2 * c * s + self.costWeights[1] * s * (1-c)
        Lx[2] = self.costWeights[3]**2 * ydot
        Lx[3] = self.costWeights[4]**2 * thdot


        Lu[0] = self.costWeights[5] * u


        Lxx[0,0] = self.costWeights[2]**2
        Lxx[1,1] = self.costWeights[0]**2 * c * c + self.costWeights[1] * s * s   # GN approximation
        Lxx[2,2] = self.costWeights[3]**2
        Lxx[3,3] = self.costWeights[4]**2

        Luu[0,] = self.costWeights[5]**2

        data.Lx = Lx
        data.Lu = Lu
        data.Lxx = Lxx
        data.Luu = Luu
        data.Lxu = Lxu


if __name__ == '__main__':

    # USE_NUMDIFF = True

    # Creating the DAM for the cartpole
    # if USE_NUMDIFF:
    #     cartpoleDAM = DifferentialActionModelCartpole()
    #     cartpoleDiff = crocoddyl.DifferentialActionModelNumDiff(cartpoleDAM, True)
    #     terminalCartpoleDAM = DifferentialActionModelCartpoleNODIFF()
    #     terminalCartpoleDiff = crocoddyl.DifferentialActionModelNumDiff(terminalCartpoleDAM, True)
    # else:
    cartpoleDAM = DifferentialActionModelCartpole()
    terminalCartpoleDAM = DifferentialActionModelCartpole()
    # Getting the IAM using the simpletic Euler rule
    timeStep = 5e-2
    cartpoleIAM = crocoddyl.IntegratedActionModelEuler(cartpoleDAM, timeStep)

    # Creating the shooting problem
    x0 = np.array([0., 0., 0., 0.])
    T = 50

    terminalCartpoleDAM.costWeights[0] = 200
    terminalCartpoleDAM.costWeights[1] = 200
    terminalCartpoleDAM.costWeights[2] = 1.
    terminalCartpoleDAM.costWeights[3] = 0.1
    terminalCartpoleDAM.costWeights[4] = 0.01
    terminalCartpoleDAM.costWeights[5] = 0.0001
    terminalCartpoleIAM = crocoddyl.IntegratedActionModelEuler(terminalCartpoleDAM, timeStep)

    problem = crocoddyl.ShootingProblem(x0, [cartpoleIAM] * T, terminalCartpoleIAM)
    # Solving it using DDP
    ddp = crocoddyl.SolverFDDP(problem)

    ddp.setCallbacks([crocoddyl.CallbackVerbose()])
    xs = [x0] * (ddp.problem.T + 1)
    us = [np.zeros(1)] * ddp.problem.T 

    ddp.solve(xs, us, 10000, False)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.plot(np.array(ddp.xs)[:, 0], label="ddp")
    ax2.plot(np.array(ddp.xs)[:, 1], label="ddp")
    ax3.plot(np.array(ddp.xs)[:, 2], label="ddp")
    ax4.plot(np.array(ddp.xs)[:, 2], label="ddp")

    ax1.set_ylabel(r"$x$")
    ax2.set_ylabel(r"theta")
    ax3.set_ylabel(r"$v_x$")
    ax4.set_ylabel(r"theta dot")



    fig, (ax1) = plt.subplots(1, 1)
    ax1.plot(np.array(ddp.us)[:, 0], label="ddp")

    ax1.set_ylabel(r"$u$")
    # plt.show()


    # plt.show()

    # Display animation
    animateCartpole(ddp.xs, show=True)