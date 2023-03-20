## Implementation of the sequential constrained lqr
## Author : Avadesh Meduri and Armand Jordana
## Date : 9/3/2022

import numpy as np
from numpy import linalg
import proxsuite

import crocoddyl
from crocoddyl import SolverAbstract
import scipy.linalg as scl
import osqp
from clqr import CLQR

def rev_enumerate(l):
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error

class CILQR(CLQR):

    def __init__(self, shootingProblem, constraintModel, method="ProxQP"):
        CLQR.__init__(self, shootingProblem, constraintModel, method=method)
        self.mu = 1e1
    
    def models(self):
        mod = [m for m in self.problem.runningModels]
        mod += [self.problem.terminalModel]
        return mod      


    def tryStep(self, alpha):
        """
        This function tries the step 
        """

        self.merit_try = 0
        self.cost_try = 0
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.xs_try[t] = model.state.integrate(self.xs[t], alpha*self.dx[t])
            self.us_try[t] = self.us[t] + alpha*self.du[t]    
        self.xs_try[-1] = model.state.integrate(self.xs[-1], alpha*self.dx[-1]) ## terminal state update


        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            model.calc(data, self.xs_try[t], self.us_try[t])  
            self.gap_try[t] = model.state.diff(self.xs_try[t+1], data.xnext) #gaps
            self.cost_try += data.cost

        self.problem.terminalModel.calc(self.problem.terminalData, self.xs_try[-1])
        self.cost_try += self.problem.terminalData.cost

        self.gap_norm_try = sum(np.linalg.norm(self.gap_try, 1, axis = 1))

        self.merit_try = self.cost_try + self.mu*self.gap_norm_try



    def solve(self, init_xs=None, init_us=None, maxiter=100, isFeasible=False, regInit=None):
        #___________________ Initialize ___________________#
        if init_xs is None or len(init_xs) < 1:
            init_xs = [self.problem.x0.copy() for m in self.models()] 
        if init_us is None or len(init_us) < 1:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels] 

        init_xs[0][:] = self.problem.x0.copy() # Initial condition guess must be x0
        self.setCandidate(init_xs, init_us, False)

        alpha = None
        for i in range(maxiter):
            self.computeDirection()

            self.merit =  self.cost + self.mu*self.gap_norm
            print("iter", i, "merit", self.merit, "cost", self.cost, "gap norms", self.gap_norm, "dx norm", self.x_grad_norm, "du norm", self.u_grad_norm, "alpha", alpha)

            alpha = 1.
            self.tryStep(alpha)
            max_search = 20
            for k in range(max_search):
                if k == max_search - 1:
                    print("No improvement")
                    return False
                # print("iter_try", k, "Total merit", self.merit_try, "Total cost", self.cost_try, "gap norms", self.gap_norm_try, "step length", alpha)

                if self.merit < self.merit_try:     # backward pass with regularization 
                    alpha *= 0.5
                    self.tryStep(alpha)
                else:
                    self.setCandidate(self.xs_try, self.us_try, False)
                    break
        
            if self.x_grad_norm < 1e-5 and self.u_grad_norm < 1e-5:
                print("Converged")
                break

        print("Final :", " merit", self.merit, "cost", self.cost, "gap norms", self.gap_norm, "dx norm", self.x_grad_norm, "du norm", self.u_grad_norm, "alpha", alpha)


    def allocateData(self):
        self.xs_try = [np.zeros(m.state.nx) for m in self.models()] 
        self.xs_try[0][:] = self.problem.x0.copy()
        self.us_try = [np.zeros(m.nu) for m in self.problem.runningModels] 
        #
        self.dx = [np.zeros(m.state.ndx) for m  in self.models()]
        self.du = [np.zeros(m.nu) for m  in self.problem.runningModels] 
        # 
        self.lxmin = self.constraintModel[0]
        self.lxmax = self.constraintModel[1]
        self.lumin = self.constraintModel[2]
        self.lumax = self.constraintModel[3]
        #
        self.S = [np.zeros([m.state.ndx, m.state.ndx]) for m in self.models()]   
        self.s = [np.zeros(m.state.ndx) for m in self.models()]   
        self.L = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        self.l = [np.zeros([m.nu]) for m in self.problem.runningModels]
        #
        self.x_grad = [np.zeros(m.state.ndx) for m in self.models()]
        self.u_grad = [np.zeros(m.nu) for m in self.problem.runningModels]

        self.gap = [np.zeros(m.state.ndx) for m in self.models()] # gaps
        self.gap_try = [np.zeros(m.state.ndx) for m in self.models()] # gaps for line search

        self.x_grad_norm = 0
        self.u_grad_norm = 0
        self.gap_norm = 0
        self.cost = 0
        self.cost_try = 0
        self.merit = 0
        self.merit_old = 0
        self.nx = self.problem.terminalModel.state.nx 
        self.nu = self.problem.runningModels[0].nu
        
    def check_optimality(self):
        """
        This function checks if the convexified lqr problem reaches optimality before we take the next step of the SQP
        """
        error = 0
        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels,self.problem.runningDatas)):

            r = data.Lu
            q = data.Lx
            R = data.Luu
            Q = data.Lxx
            P = data.Lxu.T
            A = data.Fx
            B = data.Fu 

            # print(self.gap[t].shape, np.shape(self.S[t+1]))

            h = r + B.T@(self.s[t+1] + self.S[t+1]@self.gap[t])
            G = P + B.T@self.S[t+1]@A
            H = R + B.T@self.S[t+1]@B

            error += np.linalg.norm(H@self.du[t] + h + G@self.dx[t]) ## optimality check

        assert error < 1e-6