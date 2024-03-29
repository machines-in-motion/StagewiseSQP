### This is a python implementation of stagewise SQP with crocoddyl with a different line search and step
### Date : 17/02/2023
## Authors : Avadesh, Armand, Sebastien

import numpy as np
from numpy import linalg

import crocoddyl
from crocoddyl import SolverAbstract
import scipy.linalg as scl

LINE_WIDTH = 100 

VERBOSE = False    

def rev_enumerate(l):
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error

class SSQP(SolverAbstract):
    def __init__(self, shootingProblem, use_heuristic_ls=False, VERBOSE=False):
        SolverAbstract.__init__(self, shootingProblem)
        self.x_reg = 0
        self.u_reg = 0
        self.regFactor = 10
        self.regMax = 1e9
        self.regMin = 1e-9
        self.mu = 1e0
        self.termination_tolerance = 1e-8
        self.use_heuristic_ls = use_heuristic_ls

        self.VERBOSE = VERBOSE
        
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
            # model.calc(data, self.xs[t], self.us[t])  
            self.gap[t] = model.state.diff(self.xs[t+1], data.xnext) #gaps
            self.cost += data.cost
        
        # self.gap_norm = np.linalg.norm(self.gap, 1)
        self.gap_norm = sum(np.linalg.norm(self.gap, 1, axis = 1)) / self.problem.T

        self.cost += self.problem.terminalData.cost 
        self.merit =  self.cost + self.mu*self.gap_norm

    def computeDirection(self, recalc=True):
        self.calc()
        self.KKT_check()
        self.backwardPass()  
        self.computeUpdates()

        # self.compute_expected_decrease()

        # self.safety_check()

    def LQ_problem_KKT_check(self):
        KKT = 0
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            KKT += max(abs(data.Lxx @ self.dx[t] + data.Lxu @ self.du[t] + data.Lx + data.Fx.T @ self.lag_mul[t+1] - self.lag_mul[t]))
            KKT += max(abs(data.Luu @ self.du[t] + data.Lxu.T @ self.dx[t] + data.Lu + data.Fu.T @ self.lag_mul[t+1]))

        KKT += max(abs(self.problem.terminalData.Lxx @ self.dx[-1] + self.problem.terminalData.Lx - self.lag_mul[-1]))
        # print("\n THIS SHOULD BE ZERO ", KKT)

        # print("\n")

    def KKT_check(self):
        self.KKT = 0
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.KKT = max(self.KKT, max(abs(data.Lx + data.Fx.T @ self.lag_mul[t+1] - self.lag_mul[t])))
            self.KKT = max(self.KKT, max(abs(data.Lu + data.Fu.T @ self.lag_mul[t+1])))

        self.KKT = max(self.KKT, max(abs(self.problem.terminalData.Lx - self.lag_mul[-1])))
        self.KKT = max(self.KKT, max(abs(np.array(self.fs).flatten())))
        if(self.VERBOSE):
            print("\nInfinity norm of KKT condition ", self.KKT)
            print("\n")



    def computeUpdates(self): 
        """ computes step updates dx and du """
        self.expected_decrease = 0
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
                # here we compute the direction 
                self.lag_mul[t] = self.S[t] @ self.dx[t] + self.s[t]
                self.du[t][:] = self.L[t].dot(self.dx[t]) + self.l[t] 
                A = data.Fx
                B = data.Fu      
                if len(data.Fu.shape) == 1:
                    bl = B.dot(self.l[t][0])
                    BL = B.reshape(B.shape[0], 1)@self.L[t]
                else: 
                    bl = B @ self.l[t]
                    BL = B@self.L[t]
                self.dx[t+1] = (A + BL)@self.dx[t] + bl + self.gap[t]  

        self.lag_mul[-1] = self.S[-1] @ self.dx[-1] + self.s[-1]
        self.x_grad_norm = np.linalg.norm(self.dx)/self.problem.T
        self.u_grad_norm = np.linalg.norm(self.du)/self.problem.T
        # print("x_norm", self.x_grad_norm,"u_norm", self.u_grad_norm )


    def compute_expected_decrease(self):

        self.expected_decrease = 0
        self.rho = 0.999999
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            q = data.Lx    
            r = data.Lu
            self.expected_decrease += q.T@self.dx[t] + r.T@self.du[t]
            hess_decrease = self.du[t].T@data.Luu@self.du[t] + self.dx[t].T@data.Lxx@self.dx[t]
            # if hess_decrease > 0:
            #     self.expected_decrease += 0.5*hess_decrease
        
        q = self.problem.terminalData.Lx
        self.expected_decrease += q.T@self.dx[-1] 
        
        hess_decrease = self.dx[-1].T@data.Lxx@self.dx[-1]
        # if hess_decrease > 0:
        #     self.expected_decrease += 0.5*hess_decrease
        tmp_mu = self.expected_decrease/((1 - self.rho)*self.gap_norm)
        if(self.VERBOSE):
            print(self.expected_decrease, (1 - self.rho)*self.gap_norm)
            print(tmp_mu)
        # self.mu = tmp_mu

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

        self.gap_norm_try = np.linalg.norm(self.gap_try, 1) / self.problem.T

        self.merit_try = self.cost_try + self.mu*self.gap_norm_try


    def acceptStep(self):

        self.setCandidate(self.xs_try, self.us_try, False)

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
            # self.H += 1e-9 * np.eye(len(self.H))
            if len(G.shape) == 1:
                G = np.resize(G,(1,G.shape[0]))
            ## Making sure H is PD

            # print(H.shape, R.shape, B.shape, G.shape)
            Lb_uu = scl.cho_factor(self.H, lower=True)
            # while True:
            #     try:
            #         Lb_uu = scl.cho_factor(self.H, lower=True)
            #         break 
            #     except:
            #         print("increasing H")
            #         self.H += 100*self.regMin*np.eye(len(self.H))

            H = self.H.copy()
            self.L[t][:,:] = -1*scl.cho_solve(Lb_uu, G)
            self.l[t][:] = -1*scl.cho_solve(Lb_uu, h)
            
            self.S[t] = Q + A.T @ (self.S[t+1])@A - self.L[t].T@H@self.L[t] 
            self.S[t] = (self.S[t] + self.S[t].T) / 2 
            self.s[t] = q + A.T @ (self.S[t+1] @ self.gap[t] + self.s[t+1]) + G.T@self.l[t][:]+ self.L[t].T@(h + H@self.l[t][:])

    def solve(self, init_xs=None, init_us=None, maxiter=100, isFeasible=False, regInit=None):
        #___________________ Initialize ___________________#
        if init_xs is None or len(init_xs) < 1:
            init_xs = [self.problem.x0.copy() for m in self.models()] 
        if init_us is None or len(init_us) < 1:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels] 

        init_xs[0][:] = self.problem.x0.copy() # Initial condition guess must be x0
        
        self.setCandidate(init_xs, init_us, False)
        self.calc() # compute the gaps 
        alpha = None
        for i in range(maxiter):
            recalc = True   # this will recalculated derivatives in Compute Direction 
            self.computeDirection(recalc=recalc)
            if(self.VERBOSE):
                print("iter", i, "Total merit", self.merit, "Total cost", self.cost, "gap norms", self.gap_norm, "step length", alpha)

            alpha = 1.
            self.tryStep(alpha)
            max_search = 20
            for k in range(max_search):
                if k == max_search - 1:
                    print("No improvement")
                    return False
                if self.use_heuristic_ls:
                    if self.gap_norm < self.gap_norm_try and self.cost < self.cost_try:
                        alpha *= 0.5
                        self.tryStep(alpha)
                    else:
                        self.acceptStep()
                        break
                else:
                    if self.merit < self.merit_try:
                        alpha *= 0.5
                        self.tryStep(alpha)
                    else:
                        self.acceptStep()
                        break

            # self.check_optimality()
            self.calc()

            # print("grad norm", self.x_grad_norm + self.u_grad_norm)
            # if abs(self.merit - self.merit_old) < 1e-4:
            # if self.x_grad_norm + self.u_grad_norm < 1e-4:
            if self.KKT < self.termination_tolerance:
                if(self.VERBOSE):
                    print("KKT condition reached")
                    print("Terminated", "Total merit", self.merit, "Total cost", self.cost, "gap norms", self.gap_norm, "step length", alpha)
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
        self.lag_mul = [np.zeros(m.state.ndx) for m  in self.models()] 
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