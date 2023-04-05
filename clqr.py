## This is the implementation of the constrained LQR
## Author : Avadesh Meduri and Armand Jordana
## Date : 8/03/2023

import numpy as np
from crocoddyl import SolverAbstract
import scipy.linalg as scl
from qpsolvers import QPSolvers
from py_osqp import CustomOSQP
from scipy.sparse.linalg import spsolve

LINE_WIDTH = 100 

VERBOSE = False    
pp = lambda s : np.format_float_scientific(s, exp_digits=2, precision =2)

def rev_enumerate(l):
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error


class CLQR(SolverAbstract, QPSolvers, CustomOSQP):
    def __init__(self, shootingProblem, constraintModel, method):
        SolverAbstract.__init__(self, shootingProblem)
        
        self.sigma = 1e-6
        self.rho_sparse= 1e-1
        self.rho_min = 1e-6
        self.rho_max = 1e6
        self.alpha = 1.6

        self.eps_abs = 1e-3
        self.eps_rel = 1e-3
        self.adaptive_rho_tolerance = 5
        self.rho_update_interval = 25
        self.eps_abs = 1e-3
        self.eps_rel = 1e-3
        self.regMin = 1e-6

        self.constraintModel = constraintModel
        self.allocateData()
        self.allocateQPData()
        
        QPSolvers.__init__(self, method)
        CustomOSQP.__init__(self)
        
        self.max_iters = 5000

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
        if self.method == "ProxQP" or self.method=="OSQP" or self.method == "CustomOSQP" or self.method == "Boyd":
            self.computeDirectionFullQP()
        else:
            maxit = 100

            # QPSolvers.__init__(self, "Boyd")
            # res = self.computeDirectionFullQP(maxit)

            # self.dx_test[0] = np.zeros(self.nx)
            # for t in range(self.problem.T):
            #     self.dx_test[t+1] = res[t * self.nx: (t+1) * self.nx] 
            #     index_u = self.problem.T*self.nx + t * self.nu
            #     self.du_test[t] = res[index_u:index_u+self.nu]
                
            self.calc(True)
            for iter in range(1, maxit+1):
                
                self.backwardPass()  
                self.computeUpdates()
                self.update_lagrangian_parameters_infinity()
                # for i in range(len(self.dx)):
                #     print(self.dx_tilde_test[i], self.dx_test[i])
                self.update_rho_sparse(iter)

                if (iter) % self.rho_update_interval == 0 and iter > 1:
                    print("Iters", iter, "res-primal", pp(self.norm_primal), "res-dual", pp(self.norm_dual)\
                                    , "optimal rho estimate", pp(self.rho_estimate_sparse), "rho", pp(self.rho_sparse), "\n") 
            
                if self.norm_primal < self.eps_abs + self.eps_rel*self.norm_primal_rel and\
                        self.norm_dual < self.eps_abs + self.eps_rel*self.norm_dual_rel:
                            print("QP converged")
                            break

            print("Norm linalg Dx", np.linalg.norm(np.array(self.dx) - np.array(self.dx_test)))
            print("Norm linalg Du", np.linalg.norm(np.array(self.du) - np.array(self.du_test)))

            # print("\n")
                
    def update_rho_sparse(self, iter):
        scale = np.sqrt(self.norm_primal * self.norm_dual_rel/(self.norm_dual * self.norm_primal_rel))
        self.rho_estimate_sparse = scale * self.rho_sparse
        self.rho_estimate_sparse = min(max(self.rho_estimate_sparse, self.rho_min), self.rho_max) 

        if (iter) % self.rho_update_interval == 0 and iter > 1:
            if self.rho_estimate_sparse > self.rho_sparse* self.adaptive_rho_tolerance or\
                self.rho_estimate_sparse < self.rho_sparse/ self.adaptive_rho_tolerance :
                self.rho_sparse= self.rho_estimate_sparse
                for t, cmodel in enumerate(self.constraintModel):   
                    for k in range(cmodel.ncx):  
                        if cmodel.lxmin[k] == -np.inf and cmodel.lxmax[k] == np.inf:
                            self.rho_vec_x[t][k] = self.rho_min 
                        elif cmodel.lxmin[k] == cmodel.lxmax[k]:
                            self.rho_vec_x[t][k] = 1e3 * self.rho_sparse
                        elif cmodel.lxmin[k] != cmodel.lxmax[k]:
                            self.rho_vec_x[t][k] = self.rho_sparse
                    for k in range(cmodel.ncu):  
                        if cmodel.lumin[k] == -np.inf and cmodel.lumax[k] == np.inf:
                            self.rho_vec_u[t][k] = self.rho_min 
                        elif cmodel.lumin[k] == cmodel.lumax[k]:
                            self.rho_vec_u[t][k] = 1e3 * self.rho_sparse
                        elif cmodel.lumin[k] != cmodel.lumax[k]:
                            self.rho_vec_u[t][k] = self.rho_sparse

    def update_lagrangian_parameters_infinity(self):

        self.norm_primal = -np.inf
        self.norm_dual = -np.inf
        self.norm_primal_rel, self.norm_dual_rel = [-np.inf,-np.inf], -np.inf
        
        for t, (model, data, cmodel) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas, self.constraintModel[:-1])):
            cx, cu =  cmodel.calc(self.xs[t], self.us[t])
            Cx, Cu =  cmodel.calcDiff(self.xs[t], self.us[t])

            xz_k = self.xz[t].copy()
            uz_k = self.uz[t].copy()
            
            self.dx_relaxed[t] = self.alpha * Cx @ self.dx[t] + (1 - self.alpha)*self.xz[t]
  
            self.du_relaxed[t] = self.alpha * Cu @ self.du[t] + (1 - self.alpha)*self.uz[t]

            self.xz[t] = np.clip(self.dx_relaxed[t] + np.divide(self.xy[t], self.rho_vec_x[t]), cmodel.lxmin - cx, cmodel.lxmax - cx)
            self.uz[t] = np.clip(self.du_relaxed[t] + np.divide(self.uy[t], self.rho_vec_u[t]), cmodel.lumin - cu, cmodel.lumax - cu)
            
            self.xy[t] += np.multiply(self.rho_vec_x[t], (self.dx_relaxed[t] - self.xz[t])) 
            self.uy[t] += np.multiply(self.rho_vec_u[t], (self.du_relaxed[t] - self.uz[t]))

            dual_vec_x = Cx.T@ np.multiply(self.rho_vec_x[t], (self.xz[t] - xz_k))
            dual_vec_u = Cu.T@ np.multiply(self.rho_vec_u[t], (self.uz[t] - uz_k))

            self.norm_dual = max(self.norm_dual, max(abs(dual_vec_x)), max(abs(dual_vec_u)))
            self.norm_primal = max(self.norm_primal, max(abs(Cx@self.dx[t] - self.xz[t])), max(abs(Cu@self.du[t] - self.uz[t])))
            self.norm_primal_rel[0] = max(self.norm_primal_rel[0], max(abs(Cx@self.dx[t])), max(abs(Cu@self.du[t])))
            self.norm_primal_rel[1] = max(self.norm_primal_rel[1], max(abs(self.xz[t])), max(abs(self.uz[t])))
            self.norm_dual_rel = max(self.norm_dual_rel, max(abs(Cx.T@self.xy[t])), max(abs(Cu.T@self.uy[t])))


        cmodel = self.constraintModel[-1]
        cx, _ =  cmodel.calc(self.xs[-1])
        Cx, _ =  cmodel.calcDiff(self.xs[-1])

        xz_k = self.xz[-1].copy()
        self.dx_relaxed[-1] = self.alpha * Cx @ self.dx[-1] + (1 - self.alpha)*self.xz[-1]

        self.xz[-1] = np.clip(self.dx_relaxed[-1] + np.divide(self.xy[-1], self.rho_vec_x[-1]), cmodel.lxmin - cx, cmodel.lxmax - cx)
        self.xy[-1] += np.multiply(self.rho_vec_x[-1], (self.dx_relaxed[-1] - self.xz[-1])) 

        dual_vec_x = Cx.T@np.multiply(self.rho_vec_x[-1], (self.xz[-1] - xz_k))

        self.norm_dual = max(self.norm_dual, max(abs(dual_vec_x)), max(abs(dual_vec_u)))
        self.norm_primal = max(self.norm_primal, max(abs(Cx@self.dx[-1] - self.xz[-1])))
        self.norm_primal_rel[0] = max(self.norm_primal_rel[0], max(abs(Cx@self.dx[-1])))
        self.norm_primal_rel[1] = max(self.norm_primal_rel[1], max(abs(self.xz[-1])))
        self.norm_primal_rel = max(self.norm_primal_rel)
        self.norm_dual_rel = max(self.norm_dual_rel, max(abs(Cx.T@self.xy[-1])))

    def computeUpdates(self): 
        """ computes step updates dx and du """
        self.expected_decrease = 0
        self.dx_old[0] = self.dx[0].copy()
        assert np.linalg.norm(self.dx[0]) < 1e-6
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
        Cx, _ =  self.constraintModel[-1].calcDiff(self.xs[-1])
        rho_mat_x = self.rho_vec_x[-1]*np.eye(len(self.rho_vec_x[-1]))

        self.S[-1][:,:] = self.problem.terminalData.Lxx + self.sigma*np.eye(self.problem.terminalModel.state.nx) \
                                                        + (Cx.T @ rho_mat_x @ Cx)
        self.s[-1][:] = self.problem.terminalData.Lx +  Cx.T@rho_mat_x@( np.divide(self.xy[-1],self.rho_vec_x[-1]) - self.xz[-1])[:] \
                                                    + (- self.sigma * self.dx_old[-1])
        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels,self.problem.runningDatas)):

            Cx, Cu =  self.constraintModel[t].calcDiff(self.xs[t], self.us[t])
            rho_mat_x = self.rho_vec_x[t]*np.eye(len(self.rho_vec_x[t]))
            rho_mat_u = self.rho_vec_u[t]*np.eye(len(self.rho_vec_u[t]))

            r = data.Lu + Cu.T@rho_mat_u@(np.divide(self.uy[t],self.rho_vec_u[t]) - self.uz[t])[:] + (- self.sigma * self.du_old[t])
            q = data.Lx + Cx.T@rho_mat_x@(np.divide(self.xy[t],self.rho_vec_x[t]) - self.xz[t])[:] + (- self.sigma * self.dx_old[t])
            R = data.Luu + self.sigma*np.eye(model.nu) + (Cu.T @ rho_mat_u @ Cu)
            Q = data.Lxx + self.sigma*np.eye(model.state.nx) + (Cx.T @ rho_mat_x @ Cx)
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
        
    def allocateQPData(self):
        self.xz = [np.zeros(cmodel.ncx) for cmodel in self.constraintModel]
        self.uz = [np.zeros(cmodel.ncu) for cmodel in self.constraintModel] 
        self.xy = [np.zeros(cmodel.ncx) for cmodel in self.constraintModel]
        self.uy = [np.zeros(cmodel.ncu) for cmodel in self.constraintModel] 

        self.xz_test = [np.zeros(cmodel.ncx) for cmodel in self.constraintModel]
        self.uz_test = [np.zeros(cmodel.ncu) for cmodel in self.constraintModel] 
        self.xy_test = [np.zeros(cmodel.ncx) for cmodel in self.constraintModel]
        self.uy_test = [np.zeros(cmodel.ncu) for cmodel in self.constraintModel] 


        self.rho_vec_x = [np.zeros(cmodel.ncx) for cmodel in self.constraintModel]
        self.rho_vec_u = [np.zeros(cmodel.ncu) for cmodel in self.constraintModel]
        self.rho_estimate_sparse = 0.0

        for t, cmodel in enumerate(self.constraintModel):   
            for k in range(cmodel.ncx):  
                if cmodel.lxmin[k] == -np.inf and cmodel.lxmax[k] == np.inf:
                    self.rho_vec_x[t][k] = self.rho_min 
                elif cmodel.lxmin[k] == cmodel.lxmax[k]:
                    self.rho_vec_x[t][k] = 1e3 * self.rho_sparse
                elif cmodel.lxmin[k] != cmodel.lxmax[k]:
                    self.rho_vec_x[t][k] = self.rho_sparse
            for k in range(cmodel.ncu):  
                if cmodel.lumin[k] == -np.inf and cmodel.lumax[k] == np.inf:
                    self.rho_vec_u[t][k] = self.rho_min 
                elif cmodel.lumin[k] == cmodel.lumax[k]:
                    self.rho_vec_u[t][k] = 1e3 * self.rho_sparse
                elif cmodel.lumin[k] != cmodel.lumax[k]:
                    self.rho_vec_u[t][k] = self.rho_sparse

    def allocateData(self):
        self.xs_try = [np.zeros(m.state.nx) for m in self.models()] 
        self.xs_try[0][:] = self.problem.x0.copy()
        self.us_try = [np.zeros(m.nu) for m in self.problem.runningModels] 
        # 
        self.dx = [np.zeros(m.state.ndx) for m  in self.models()]
        self.du = [np.zeros(m.nu) for m  in self.problem.runningModels] 
        self.dx_relaxed = [np.zeros(m.state.ndx) for m  in self.models()]
        self.du_relaxed = [np.zeros(m.nu) for m  in self.problem.runningModels] 
        self.dx_old = [np.zeros(m.state.ndx) for m  in self.models()]
        self.du_old = [np.zeros(m.nu) for m  in self.problem.runningModels] 

        self.dx_test = [np.zeros(m.state.ndx) for m  in self.models()]
        self.du_test = [np.zeros(m.nu) for m  in self.problem.runningModels] 
        
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
