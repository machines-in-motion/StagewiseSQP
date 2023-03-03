## This is  a python implementation of GNMS with an FDDP backward pass from C++ and line search and step in python. 
## This file provides the baseline to transfer code completely to C++
## Date : 17/02/2023
## Authors : Avadesh, Armand, Sebastien

import numpy as np

from crocoddyl import SolverFDDP

class GNMSCPP(SolverFDDP):

    def __init__(self, shootingProblem):
        
        SolverFDDP.__init__(self, shootingProblem)
        self.mu = 1e3

        self.allocateData()

    def calc(self):
        
        self.calcDiff()
        self.gap_norm = sum(np.linalg.norm(self.fs, 1, axis = 1))
        self.merit = self.cost + self.mu*self.gap_norm
        print(self.gap_norm)
        # print(self.cost, self.cost_try, self.gap_norm)

    def computeDirection(self):
        # print("using Python")
        self.calc()
        self.backwardPass()
        self.computeUpdates()

    def computeUpdates(self): 
        """ computes step updates dx and du """
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
                # here we compute the direction 
                self.du[t][:] = -self.K[t].dot(self.dx[t]) - self.k[t] 
                A = data.Fx
                B = data.Fu      
                if len(data.Fu.shape) == 1:
                    bl = -B.dot(self.k[t][0])
                    BL = -B.reshape(B.shape[0], 1)@ self.K[t].reshape(1, B.shape[0])
                else: 
                    bl = -B @ self.k[t]
                    BL = -B@ self.K[t]
                self.dx[t+1] = (A + BL)@self.dx[t] + bl + self.fs[t+1]  

        self.x_grad_norm = sum(np.linalg.norm(self.dx, axis = 1))/self.problem.T
        self.u_grad_norm = sum(np.linalg.norm(self.du, axis = 1))/self.problem.T
        print("x_norm", self.x_grad_norm,"u_norm", self.u_grad_norm )

    def tryStep(self, alpha):
        print("using python")
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

    def acceptStep(self):

        self.setCandidate(self.xs_try, self.us_try, False)


    def solve(self, init_xs=None, init_us=None, maxiter=100, isFeasible=False, regInit=None):

        if init_xs is None or len(init_xs) < 1:
            init_xs = [np.zeros(m.state.nx) for m in self.models()] 
        if init_us is None or len(init_us) < 1:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels] 

        init_xs[0][:] = self.problem.x0.copy() # Initial condition guess must be x0
        
        self.setCandidate(init_xs, init_us, False)
        alpha = None
        self.computeDirection()
        self.tryStep(1.0)
        print(self.gap_norm, self.cost)

        # assert False
        for i in range(maxiter):
            recalc = True   # this will recalculated derivatives in Compute Direction 
            print("iter", i, "Total merit", self.merit, "Total cost", self.cost, "gap norms", self.gap_norm, "step length", alpha)

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
                    self.acceptStep()
                    break

            # self.check_optimality()
            self.computeDirection()

            # print("grad norm", self.x_grad_norm + self.u_grad_norm)
            # if abs(self.merit - self.merit_old) < 1e-4:
            if self.x_grad_norm + self.u_grad_norm < 1e-4:
                print("No improvement observed")
                print("Terminated", "Total merit", self.merit, "Total cost", self.cost, "gap norms", self.gap_norm, "step length", alpha)
                break

        return True 

    def models(self):
        mod = [m for m in self.problem.runningModels]
        mod += [self.problem.terminalModel]
        return mod  

    def allocateData(self):
        self.xs_try = [np.zeros(m.state.nx) for m in self.models()] 
        self.xs_try[0][:] = self.problem.x0.copy()
        self.us_try = [np.zeros(m.nu) for m in self.problem.runningModels] 
        # 
        self.dx = [np.zeros(m.state.ndx) for m  in self.models()]
        self.du = [np.zeros(m.nu) for m  in self.problem.runningModels] 

        self.gap = [np.zeros(m.state.nx) for m in self.models()] # gaps
        self.gap_try = [np.zeros(m.state.nx) for m in self.models()] # gaps for line search


        self.merit = 0
        self.merit_old = 0
        self.x_grad_norm = 0
        self.u_grad_norm = 0
        self.gap_norm = 0
        self.cost = 0
        self.cost_try = 0
        self.expected_decrease = 0



        