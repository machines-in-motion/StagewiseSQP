## Implementation of the sequential constrained lqr
## Author : Avadesh Meduri and Armand Jordana
## Date : 9/3/2022

import numpy as np
import scipy.linalg as scl
from . fadmm import FADMM
from .qpsolvers import QPSolvers

pp = lambda s : np.format_float_scientific(s, exp_digits=2, precision =4)

def rev_enumerate(l):
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error

class SQPOCP(FADMM, QPSolvers):

    def __init__(self, shootingProblem, constraintModel, method, verboseQP = False, verbose = False):
        self.verbose = verbose
        if method == "FADMM":
            FADMM.__init__(self, shootingProblem, constraintModel, verboseQP = verboseQP)
            self.using_qp = 0        
        else:
            QPSolvers.__init__(self, shootingProblem, constraintModel, method, verboseQP = verboseQP)
            self.using_qp = 1        

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

        if self.verbose:
            print("{: >15} {: >15} {: >15} {: >15} {: >15} {: >15} {: >15} {: >15}".format(*["iter", "merit", "cost", "gap norms", "QP iter ", "dx norm", "du norm", "alpha"]))

        alpha = None
        for i in range(maxiter):
            if self.using_qp:
                self.computeDirectionFullQP()
            else:
                self.computeDirection()

            self.merit =  self.cost + self.mu*self.gap_norm
            if self.verbose:
                print("{: >15} {: >15} {: >15} {: >15} {: >15} {: >15} {: >15} {: >15}".format(*[i, pp(self.merit), pp(self.cost), pp(self.gap_norm), self.QP_iter, pp(self.x_grad_norm), pp(self.u_grad_norm), str(alpha)]))

            alpha = 1.
            self.tryStep(alpha)
            max_search = 20
            for k in range(max_search):
                if k == max_search - 1:
                    print("No improvement")
                    return False
                # print("iter_try", k, "Total merit", self.merit_try, "Total cost", self.cost_try, "gap norms", self.gap_norm_try, "step length", alpha)

                if self.cost < self.cost_try and self.gap_norm < self.gap_norm_try:     # backward pass with regularization 
                    alpha *= 0.5
                    self.tryStep(alpha)
                else:
                    self.setCandidate(self.xs_try, self.us_try, False)
                    break
        
            if self.x_grad_norm < 1e-5 and self.u_grad_norm < 1e-5:
                if self.verbose:
                    print("Converged")
                break
        if self.verbose:
            self.calc()
            print("{: >15} {: >15} {: >15} {: >15} {: >15} {: >15} {: >15} {: >15}".format(*["Final", pp(self.merit), pp(self.cost), pp(self.gap_norm), str(None), str(None), str(None), str(None)]))
    