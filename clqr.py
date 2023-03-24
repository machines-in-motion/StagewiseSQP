## This is the implementation of the constrained LQR
## Author : Avadesh Meduri and Armand Jordana
## Date : 8/03/2023

import numpy as np
from crocoddyl import SolverAbstract
import scipy.linalg as scl
from qpsolvers import QPSolvers


LINE_WIDTH = 100 

VERBOSE = False    

def rev_enumerate(l):
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error


class CLQR(SolverAbstract, QPSolvers):
    def __init__(self, shootingProblem, constraintModel, method):
        SolverAbstract.__init__(self, shootingProblem)
        
        self.rho_op = 1e-2
        self.sigma = 1e-6
        self.alpha = 1.6
        self.eps_abs = 1e-3
        self.eps_rel = 1e-3
        self.regMin = 1e-6
        self.constraintModel = constraintModel
        self.allocateData()
        self.allocateQPData()
        
        QPSolvers.__init__(self, method)
        
        self.max_iters = 10000

    def models(self):
        mod = [m for m in self.problem.runningModels]
        mod += [self.problem.terminalModel]
        return mod  
        
    def calc(self, recalc = True):
        # compute cost and derivatives at deterministic nonlinear trajectory 
        if recalc:
            self.problem.calc(self.xs, self.us)
            self.problem.calcDiff(self.xs, self.us)
        self.cost = 0
        # self.merit_old = self.merit        

        for t, (model, data) in enumerate(zip(self.problem.runningModels,self.problem.runningDatas)):
            # model.calc(data, self.xs[t], self.us[t])  
            self.gap[t] = model.state.diff(self.xs[t+1], data.xnext) #gaps
            self.cost += data.cost
        
        self.gap_norm = sum(np.linalg.norm(self.gap, 1, axis = 1))

        self.cost += self.problem.terminalData.cost 

    def computeDirection(self):
        if self.method == "ProxQP" or self.method=="OSQP":
            self.computeDirectionFullQP()
        else:
            self.calc(True)
            self.rho = self.rho_op
            for i in range(self.max_iters):
                
                self.backwardPass()  
                self.computeUpdates()
                self.update_lagrangian_parameters()
                # print(np.sqrt((self.norm_primal*self.norm_dual_rel)/(self.norm_dual*self.norm_primal_rel)))
                if self.norm_primal < self.eps_abs + self.eps_rel*self.norm_primal_rel and\
                   self.norm_dual < self.eps_abs + self.eps_rel*self.norm_dual_rel:
                # if self.norm_primal < self.eps_abs and\
                #    self.norm_dual < self.eps_abs:
                    print("QP converged")
                    print("Final iter ", i, " primal_residual ", self.norm_primal, " dual_residual ", self.norm_dual,\
                            "optimal rho", self.rho)
                    break

                self.rho *= np.sqrt((self.norm_primal*self.norm_dual_rel)/(self.norm_dual*self.norm_primal_rel))
                # self.rho *= np.sqrt((self.norm_primal)/(self.norm_dual))

                if i%100 == 0:
                    print("iter ", i, " primal_residual ", self.norm_primal, " dual_residual ", self.norm_dual ,  " rho", self.rho)

    def update_lagrangian_parameters(self):

        self.norm_primal = 0
        self.norm_dual = 0
        self.norm_primal_rel, self.norm_dual_rel = [0,0], 0
        
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            xz_old = self.xz[t]
            uz_old = self.uz[t]


            self.xz[t] = np.clip(self.alpha*self.Cx[t]@self.dx[t] + (1 - self.alpha)*xz_old + self.xy[t]/self.rho,\
                                    self.lxmin[t] - self.xs[t], self.lxmax[t] - self.xs[t])
            self.uz[t] = np.clip(self.alpha*self.Cu[t]@self.du[t] + (1 - self.alpha)*uz_old + self.uy[t]/self.rho,\
                                self.lumin[t] - self.us[t], self.lumax[t] - self.us[t])
            
            self.xy[t] += self.rho*(self.alpha*self.Cx[t]@self.dx[t] + (1 - self.alpha)*xz_old - self.xz[t] )
            self.uy[t] += self.rho*(self.alpha*self.Cu[t]@self.du[t] + (1 - self.alpha)*uz_old - self.uz[t] )

            ## OSQP alpha step
            self.dx[t] = self.alpha*self.dx[t] + (1- self.alpha)*self.dx_old[t]
            self.du[t] = self.alpha*self.du[t] + (1- self.alpha)*self.du_old[t]

            self.norm_dual = np.linalg.norm(self.Cx[t].T@(self.xz[t] - xz_old)) + np.linalg.norm(self.Cu[t].T@(self.uz[t] - uz_old))
            self.norm_primal = np.linalg.norm(self.xz[t] - self.Cx[t]@self.dx[t]) + np.linalg.norm(self.uz[t] - self.Cu[t]@self.du[t])
            self.norm_primal_rel[0] += np.linalg.norm(self.Cx[t]@self.dx[t]) + np.linalg.norm(self.Cu[t]@self.du[t])
            self.norm_primal_rel[1] += np.linalg.norm(self.xz[t]) + np.linalg.norm(self.uz[t])
            self.norm_dual_rel = np.linalg.norm(self.Cx[t].T@self.xy[t]) + np.linalg.norm(self.Cu[t].T@self.uy[t])

        xz_old = self.xz[-1]
    
        self.xz[-1] = np.clip(self.alpha*self.Cx[-1]@self.dx[-1] + (1 - self.alpha)*xz_old + self.xy[-1]/self.rho, \
                                self.lxmin[-1] - self.xs[-1], self.lxmax[-1] - self.xs[-1])
        self.xy[-1] += self.rho*(self.alpha*self.Cx[-1]@self.dx[-1] + (1 - self.alpha)*xz_old - self.xz[-1])
        self.dx[-1] = self.alpha*self.dx[-1] + (1- self.alpha)*self.dx_old[-1]

        self.norm_dual = np.linalg.norm(self.Cx[-1].T@(self.xz[-1] - xz_old))
        self.norm_dual *= self.rho
        self.norm_primal = np.linalg.norm(self.xz[-1] - self.Cx[t]@self.dx[-1])
        self.norm_primal_rel[0] += np.linalg.norm(self.Cx[-1]@self.dx[-1]) + np.linalg.norm(self.Cu[-1]@self.du[-1])
        self.norm_primal_rel = max(self.norm_primal_rel)
        self.norm_dual_rel = np.linalg.norm(self.Cx[t].T@self.xy[t])


    def computeUpdates(self): 
        """ computes step updates dx and du """
        self.expected_decrease = 0
        self.dx_old[0] = self.dx[0].copy()
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
                # here we compute the direction 
                self.du_old[t] = self.du[t].copy()
                self.dx_old[t+1] = self.dx[t+1].copy()

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

        self.x_grad_norm = np.linalg.norm(self.dx)/(self.problem.T+1)
        self.u_grad_norm = np.linalg.norm(self.du)/self.problem.T

        # print("x_norm", self.x_grad_norm,"u_norm", self.u_grad_norm )
                
    def acceptStep(self, alpha):
        
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.xs_try[t] = model.state.integrate(self.xs[t], alpha*self.dx[t])
            self.us_try[t] = self.us[t] + alpha*self.du[t]    
        self.xs_try[-1] = model.state.integrate(self.xs[-1], alpha*self.dx[-1]) ## terminal state update

        self.setCandidate(self.xs_try, self.us_try, False)

    def backwardPass(self): 
        self.S[-1][:,:] = self.problem.terminalData.Lxx + self.sigma*np.eye(self.problem.terminalModel.state.nx) \
                                                        + self.rho*(self.Cx[-1].T @ self.Cx[-1])
        self.s[-1][:] = self.problem.terminalData.Lx + self.Cx[-1].T@(self.xy[-1] - self.rho*self.xz[-1])[:] \
                                                    + (- self.sigma * self.dx_old[-1])
        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels,self.problem.runningDatas)):

            r = data.Lu + self.Cu[t].T@(self.uy[t] - self.rho*self.uz[t])[:] + (- self.sigma * self.du_old[t])
            q = data.Lx + self.Cx[t].T@(self.xy[t] - self.rho*self.xz[t])[:] + ( - self.sigma * self.dx_old[t])
            R = data.Luu + self.sigma*np.eye(model.nu) + self.rho*(self.Cu[t].T @ self.Cu[t])
            Q = data.Lxx + self.sigma*np.eye(model.state.nx) + self.rho*(self.Cx[t].T @ self.Cx[t])
            P = data.Lxu.T
            A = data.Fx
            B = data.Fu 

            # print(self.gap[t].shape, np.shape(self.S[t+1]))

            h = r + B.T@(self.s[t+1] + self.S[t+1]@self.gap[t])
            G = P + B.T@self.S[t+1]@A
            self.H = R + B.T@self.S[t+1]@B
            if len(G.shape) == 1:
                G = np.resize(G,(1,G.shape[0]))
            ## Making sure H is PD

            # print(H.shape, R.shape, B.shape, G.shape)
            while True:
                try:
                    Lb_uu = scl.cho_factor(self.H, lower=True)
                    break 
                except:
                    # print("increasing H")
                    self.H += 100*self.regMin*np.eye(len(self.H))

            H = self.H.copy()
            self.L[t][:,:] = -1*scl.cho_solve(Lb_uu, G)
            self.l[t][:] = -1*scl.cho_solve(Lb_uu, h)
            
            self.S[t] = Q + A.T @ (self.S[t+1])@A - self.L[t].T@H@self.L[t] 
            self.s[t] = q + A.T @ (self.S[t+1] @ self.gap[t] + self.s[t+1]) + \
                            G.T@self.l[t][:]+ self.L[t].T@(h + H@self.l[t][:])

    def solve(self, init_xs=None, init_us=None, maxiter=100, isFeasible=False, regInit=None):
        #___________________ Initialize ___________________#
        if init_xs is None or len(init_xs) < 1:
            init_xs = [self.problem.x0.copy() for m in self.models()] 
        if init_us is None or len(init_us) < 1:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels] 

        init_xs[0][:] = self.problem.x0.copy() # Initial condition guess must be x0
        self.setCandidate(init_xs, init_us, False)
        self.computeDirection()
        self.acceptStep(alpha = 1.0)
        
        print("\nInitial ")
        print("Total cost", self.cost, "gap norms", self.gap_norm)

        print("\nStep ")
        print("dx norm", self.x_grad_norm, "du norm", self.u_grad_norm)
        self.calc(True)

        print("\nFinal ")
        print("Total cost", self.cost, "gap norms", self.gap_norm)

    def allocateQPData(self):
        self.xz = [np.zeros(m.state.nx) for m  in self.models()]
        self.uz = [np.zeros(m.nu) for m  in self.problem.runningModels] 
        self.xy = [np.zeros(m.state.nx) for m  in self.models()]
        self.uy = [np.zeros(m.nu) for m  in self.problem.runningModels] 
        self.xw = [np.zeros(m.state.nx) for m  in self.models()]
        self.uw = [np.zeros(m.nu) for m  in self.problem.runningModels] 

    def allocateData(self):
        self.xs_try = [np.zeros(m.state.nx) for m in self.models()] 
        self.xs_try[0][:] = self.problem.x0.copy()
        self.us_try = [np.zeros(m.nu) for m in self.problem.runningModels] 
        # 
        self.dx = [np.zeros(m.state.ndx) for m  in self.models()]
        self.du = [np.zeros(m.nu) for m  in self.problem.runningModels] 
        self.dx_old = [np.zeros(m.state.ndx) for m  in self.models()]
        self.du_old = [np.zeros(m.nu) for m  in self.problem.runningModels] 
        #
        self.lxmin = self.constraintModel[0]
        self.lxmax = self.constraintModel[1]
        self.lumin = self.constraintModel[2]
        self.lumax = self.constraintModel[3]
        self.Cx = self.constraintModel[4] # list of constraint matrices x
        self.Cu = self.constraintModel[5] # list Constraint matrices u
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
        # 
        

        self.nx = self.problem.terminalModel.state.nx 
        self.nu = self.problem.runningModels[0].nu
