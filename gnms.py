""" solve for full observable case 
"""
import numpy as np
from numpy import linalg

import scipy.linalg as scl
import crocoddyl
from crocoddyl import SolverAbstract

LINE_WIDTH = 100 

VERBOSE = False    

def rev_enumerate(l):
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error

class GNMS(SolverAbstract):
    def __init__(self, shootingProblem):
        SolverAbstract.__init__(self, shootingProblem)
        self.x_reg = 0
        self.u_reg = 0
        self.regFactor = 10
        self.regMax = 1e9
        self.regMin = 1e-9
        self.mu = 10000.0

        self.allocateData()

    def models(self):
        mod = [m for m in self.problem.runningModels]
        mod += [self.problem.terminalModel]
        return mod  
        
    def calc(self):
        # compute cost and derivatives at deterministic nonlinear trajectory 
        self.problem.calc(self.xs, self.us)
        self.problem.calcDiff(self.xs, self.us)
        self.cost = 0
        # self.merit_old = self.merit        

        for t, (model, data) in enumerate(zip(self.problem.runningModels,self.problem.runningDatas)):
            self.gap[t] = model.state.diff(self.xs[t+1], data.xnext) #gaps
            self.cost += data.cost
        
        # self.gap_try[-1] = model.state.diff(self.xs_try[-1], data.xnext) #gaps

        self.gap_norm = np.linalg.norm(self.gap)

        self.cost += self.problem.terminalData.cost 
        self.merit =  self.cost + self.mu*self.gap_norm


    def computeDirection(self, recalc=True):
        if recalc:
            if VERBOSE: print("Going into Calc from compute direction")
            self.calc()
        if VERBOSE: print("Going into Backward Pass from compute direction")
        self.backwardPass()  
        self.computeUpdates()

    def computeUpdates(self): 
        """ computes step updates dx and du """
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
                # here we compute the direction 
                self.du[t][:] = self.L[t].dot(self.dx[t]) + self.l[t] 
                A = data.Fx
                B = data.Fu        
                self.dx[t+1] = (A + B@self.L[t])@self.dx[t] + B@self.l[t] + self.gap[t]

        self.x_grad_norm = np.linalg.norm(self.dx)
        self.u_grad_norm = np.linalg.norm(self.du)

        
    def tryStep(self, alpha):
        """
        This function tries the step 
        """
        self.merit_try = 0
        self.cost_try = 0
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.xs_try[t] = model.state.integrate(self.xs[t], alpha*self.dx[t])
            self.us_try[t] = self.us[t] + alpha*self.du[t]    

        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            model.calc(data, self.xs_try[t], self.us_try[t])  
            self.gap_try[t] = model.state.diff(self.xs_try[t+1], data.xnext) #gaps
            self.cost_try += data.cost

        self.xs_try[-1] = model.state.integrate(self.xs_try[-1], alpha*self.dx[-1]) ## terminal state update

        self.problem.terminalModel.calc(self.problem.terminalData, self.xs_try[-1])
        self.cost_try += self.problem.terminalData.cost

        self.gap_norm_try = np.linalg.norm(self.gap_try)

        self.merit_try = self.cost_try + self.mu*self.gap_norm_try


    def acceptStep(self, alpha):

        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):

            self.xs[t] = model.state.integrate(self.xs[t], alpha*self.dx[t])
            self.us[t] = self.us[t] + alpha*self.du[t]

        self.xs[-1] = model.state.integrate(self.xs[-1], alpha*self.dx[-1]) ## terminal state update

    def backwardPass(self): 
        self.S[-1][:,:] = self.problem.terminalData.Lxx
        self.s[-1][:] = self.problem.terminalData.Lx 
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
            self.H = R + B.T@self.S[t+1]@B

            ## Making sure H is PD

            # print(H.shape, R.shape, B.shape, G.shape)
            while True:
                try:
                    Lb_uu = scl.cho_factor(self.H, lower=True)
                    break 
                except:
                    print("increasing H")
                    self.H += 100*self.regMin*np.eye(len(self.H))

            H = self.H.copy()
            self.L[t][:,:] = -1*scl.cho_solve(Lb_uu, G)
            self.l[t][:] = -1*scl.cho_solve(Lb_uu, h)
            
            # H_inv = np.linalg.inv(H + 1e-3*np.eye(len(H)))

            # self.L[t][:,:] = -1*H_inv@G
            # self.l[t][:] = -1*H_inv@h
            

            self.S[t] = Q + A.T @ (self.S[t+1])@A - self.L[t].T@H@self.L[t] 
            self.s[t] = q + A.T @ (self.S[t+1] @ self.gap[t] + self.s[t+1]) + \
                            G.T@self.l[t][:]+ self.L[t].T@(h + H@self.l[t][:])

    def solve(self, init_xs=None, init_us=None, maxiter=100, isFeasible=False, regInit=None):
        #___________________ Initialize ___________________#
        if init_xs is None or len(init_xs) < 1:
            init_xs = [np.zeros(m.state.nx) for m in self.models()] 
        if init_us is None or len(init_us) < 1:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels] 

        init_xs[0][:] = self.problem.x0.copy() # Initial condition guess must be x0
        
        self.setCandidate(init_xs, init_us, False)
        self.calc() # compute the gaps 

        for i in range(maxiter):
            recalc = True   # this will recalculated derivatives in Compute Direction 
            self.computeDirection(recalc=recalc)
            alpha = 1.0

            # print("iter", i, "Total merit", self.merit, "Total cost", self.cost, "gap norms", self.gap_norm, "step length", alpha)

            self.tryStep(alpha)
            # print("iter", i, "Total merit", self.merit_try, "Total cost", self.cost_try, "gap norms", self.gap_norm_try, "step length", alpha)


            # self.tryStep(0.5)
            # print("iter", i, "Total merit", self.merit_try, "Total cost", self.cost_try, "gap norms", self.gap_norm_try, "step length", alpha)

            max_search = 20
            for k in range(max_search):
                if k == max_search - 1:
                    print("No improvement")
                    return False
                # print(k, self.merit, self.merit_try)
                print("iter_try", k, "Total merit", self.merit_try, "Total cost", self.cost_try, "gap norms", self.gap_norm_try, "step length", alpha)
                
                if self.merit < self.merit_try:     # backward pass with regularization 
                    alpha *= 0.5
                    self.tryStep(alpha)
                else:
                    print(alpha)
                    self.acceptStep(alpha)
                    break


            # self.check_optimality()
            self.calc()

            print("iter", i, "Total merit", self.merit, "Total cost", self.cost, "gap norms", self.gap_norm, "step length", alpha)

            # if abs(self.merit - self.merit_old) < 1e-4:
            if self.x_grad_norm + self.u_grad_norm < 1e-5:
                print("No improvement observed")
                break

        return True 

    def allocateData(self):
        self.xs_try = [np.zeros(m.state.nx) for m in self.models()] 
        self.xs_try[0][:] = self.problem.x0.copy()
        self.us_try = [np.zeros(m.nu) for m in self.problem.runningModels] 
        # 
        self.dx = [np.zeros(m.state.ndx) for m  in self.models()]
        self.du = [np.zeros(m.nu) for m  in self.problem.runningModels] 
        # 
        self.S = [np.zeros([m.state.ndx, m.state.ndx]) for m in self.models()]   
        self.s = [np.zeros(m.state.ndx) for m in self.models()]   
        self.L = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        self.l = [np.zeros([m.nu]) for m in self.problem.runningModels]
        #
        self.x_grad = [np.zeros(m.state.ndx) for m in self.models()]
        self.u_grad = [np.zeros(m.nu) for m in self.problem.runningModels]

        self.gap = [np.zeros(m.state.nx) for m in self.models()] # gaps
        self.gap_try = [np.zeros(m.state.nx) for m in self.models()] # gaps for line search

        self.merit = 0
        self.merit_old = 0
        self.x_grad_norm = 0
        self.u_grad_norm = 0
        self.gap_norm = 0
        self.cost = 0
        self.cost_try = 0

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