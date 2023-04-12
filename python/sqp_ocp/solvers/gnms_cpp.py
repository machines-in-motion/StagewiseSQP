## This is  a python implementation of GNMS with an FDDP backward pass from C++ and line search and step in python. 
## This file provides the baseline to transfer code completely to C++
## Date : 17/02/2023
## Authors : Avadesh, Armand, Sebastien

import numpy as np

from crocoddyl import SolverFDDP

class GNMSCPP(SolverFDDP):

    def __init__(self, shootingProblem):
        
        SolverFDDP.__init__(self, shootingProblem)
        self.mu = 1e0

        self.allocateData()

    def calc(self):
        
        self.calcDiff()
        self.gap_norm = sum(np.linalg.norm(self.fs, 1, axis = 1))
        self.merit = self.cost + self.mu*self.gap_norm
        # print(self.gap_norm)

    def computeDirection(self, kkt_check=True):
        # print("using Python")
        self.calc()
        if kkt_check:

            self.KKT_check()
            print("KKT ", self.KKT)
            if self.KKT < 1e-8:
                print("Terminated -- KKT condition reached")
                print("Total merit", self.merit, "Total cost", self.cost, "gap norms", self.gap_norm)
                return True

        self.backwardPass()
        self.computeUpdates()
        return False


    def computeUpdates(self): 
        """ computes step updates dx and du """
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
                # here we compute the direction 
                self.lag_mul[t] = self.Vxx[t] @ self.dx[t] + self.Vx[t]
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

        self.lag_mul[-1] = self.Vxx[-1] @ self.dx[-1] + self.Vx[-1]
        self.x_grad_norm = sum(np.linalg.norm(self.dx, 1,  axis = 1))/self.problem.T
        self.u_grad_norm = sum(np.linalg.norm(self.du, 1,  axis = 1))/self.problem.T
        # print("x_norm", self.x_grad_norm, "u_norm", self.u_grad_norm )

    def KKT_check(self):
        self.KKT = 0
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.KKT = max(self.KKT, max(abs(data.Lx + data.Fx.T @ self.lag_mul[t+1] - self.lag_mul[t])))
            self.KKT = max(self.KKT, max(abs(data.Lu + data.Fu.T @ self.lag_mul[t+1])))

        self.KKT = max(self.KKT, max(abs(self.problem.terminalData.Lx - self.lag_mul[-1])))
        self.KKT = max(self.KKT, max(abs(np.array(self.fs).flatten())))

    def tryStep(self, alpha):
        # print("using python")
        self.merit_try = 0
        self.cost_try = 0

        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.xs_try[t] = model.state.integrate(self.xs[t], alpha*self.dx[t])
            self.us_try[t] = self.us[t] + alpha*self.du[t]    
        self.xs_try[-1] = model.state.integrate(self.xs[-1], alpha*self.dx[-1]) ## terminal state update

        ## TMP PROX !!!!
        for i in range(self.problem.T+1):
            self.xs_try[i][0] = np.clip(self.xs_try[i][0], -1.5, 1.5)


        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            model.calc(data, self.xs_try[t], self.us_try[t])  
            self.gap_try[t] = model.state.diff(self.xs_try[t+1], data.xnext) #gaps
            self.cost_try += data.cost

        self.problem.terminalModel.calc(self.problem.terminalData, self.xs_try[-1])
        self.cost_try += self.problem.terminalData.cost
        
        self.gap_norm_try = sum(np.linalg.norm(self.gap_try, 1, axis = 1))

        self.merit_try = self.cost_try + self.mu*self.gap_norm_try
        # print("cost_try", self.cost_try, "gaps_try", self.gap_norm_try, "merit try", self.merit_try)

    def acceptStep(self):

        self.setCandidate(self.xs_try, self.us_try, False)


    def solve(self, init_xs=None, init_us=None, maxiter=100, isFeasible=False, regInit=None):

        if init_xs is None or len(init_xs) < 1:
            init_xs = [self.problem.x0.copy() for m in self.models()] 
        if init_us is None or len(init_us) < 1:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels] 

        init_xs[0][:] = self.problem.x0.copy() # Initial condition guess must be x0
        
        self.setCandidate(init_xs, init_us, False)

        alpha = None
        self.computeDirection(kkt_check=False)
        print("iter", 0, "Total merit", self.merit, "Total cost", self.cost, "gap norms", self.gap_norm)

        for i in range(maxiter):
            alpha = 1.
            self.tryStep(alpha)
            max_search = 10

            for k in range(max_search):
                if k == max_search - 1:
                    print("No improvement")
                    print("Terminated", "Total merit", self.merit, "Total cost", self.cost, "gap norms", self.gap_norm, "step length", alpha)
                    return False

                if self.gap_norm < self.gap_norm_try and self.cost < self.cost_try:
                    alpha *= 0.5
                    # print(alpha)
                    self.tryStep(alpha)
                else:
                    self.acceptStep()
                    break

            converged = self.computeDirection()

            if converged:
                return False

            print("iter", i+1,"Total merit", self.merit, "Total cost", self.cost, "gap norms", self.gap_norm, "step length", alpha)
            
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

        self.lag_mul = [np.zeros(m.state.ndx) for m  in self.models()] 

        self.gap = [np.zeros(m.state.ndx) for m in self.models()] # gaps
        self.gap_try = [np.zeros(m.state.ndx) for m in self.models()] # gaps for line search


        self.merit = 0
        self.merit_old = 0
        self.x_grad_norm = 0
        self.u_grad_norm = 0
        self.gap_norm = 0
        self.cost = 0
        self.cost_try = 0
        self.expected_decrease = 0



        