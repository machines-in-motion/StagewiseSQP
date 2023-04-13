## This contains various QP solvers that solve the convex subproblem of the sqp
## Author : Avadesh Meduri & Armand Jordana
## Date : 21/03/2022

import numpy as np
import osqp
import proxsuite
import time 
from scipy import sparse
from . py_osqp import CustomOSQP
from . fadmm_kkt import FAdmmKKT
from crocoddyl import SolverAbstract

class QPSolvers(SolverAbstract, CustomOSQP, FAdmmKKT):

    def __init__(self, shootingProblem, constraintModel, method, verboseQP = True):
        
        self.constraintModel = constraintModel
        SolverAbstract.__init__(self, shootingProblem)        

        assert method == "ProxQP" or method=="OSQP"\
              or method=="CustomOSQP" or method =="FAdmmKKT" 
        self.method = method
        if method == "CustomOSQP":
            CustomOSQP.__init__(self)
        if method == "FAdmmKKT":
            FAdmmKKT.__init__(self)

        self.allocateData()
        self.max_iters = 1000
        self.initialize = True
        self.verboseQP = verboseQP
        if self.verboseQP:
            print("USING " + str(method))

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

        for t, (cmodel, cdata, data) in enumerate(zip(self.constraintModel[:-1], self.constraintData[:-1], self.problem.runningDatas)):
            cmodel.calc(cdata, data, self.xs[t], self.us[t])
            cmodel.calcDiff(cdata, data, self.xs[t], self.us[t])

        self.constraintModel[-1].calc(self.constraintData[-1], self.problem.terminalData, self.xs[-1])
        self.constraintModel[-1].calcDiff(self.constraintData[-1], self.problem.terminalData, self.xs[-1])

        for t, (model, data) in enumerate(zip(self.problem.runningModels,self.problem.runningDatas)):
            # model.calc(data, self.xs[t], self.us[t])  
            self.gap[t] = model.state.diff(self.xs[t+1].copy(), data.xnext.copy()) #gaps
            self.cost += data.cost

        self.gap_norm = sum(np.linalg.norm(self.gap.copy(), 1, axis = 1))
        self.cost += self.problem.terminalData.cost 
        self.gap = self.gap.copy()

    def computeDirectionFullQP(self):
        self.calc(True)

        self.n_vars  = self.problem.T*(self.nx + self.nu)

        P = np.zeros((self.problem.T*(self.nx + self.nu), self.problem.T*(self.nx + self.nu)))
        q = np.zeros(self.problem.T*(self.nx + self.nu))
        
        Asize = self.problem.T*(self.nx + self.nu)
        A = np.zeros((self.problem.T*self.nx, Asize))
        B = np.zeros(self.problem.T*self.nx)

        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            index_u = self.problem.T*self.nx + t * self.nu
            if t>=1:
                index_x = (t-1) * self.nx
                P[index_x:index_x+self.nx, index_x:index_x+self.nx] = data.Lxx.copy()
                P[index_x:index_x+self.nx, index_u:index_u+self.nu] = data.Lxu.copy()
                P[index_u:index_u+self.nu, index_x:index_x+self.nx] = data.Lxu.T.copy()
                q[index_x:index_x+self.nx] = data.Lx.copy()

            P[index_u:index_u+self.nu, index_u:index_u+self.nu] = data.Luu.copy()
            q[index_u:index_u+self.nu] = data.Lu.copy()
            
            A[t * self.nx: (t+1) * self.nx, index_u:index_u+self.nu] = - data.Fu.copy() 
            A[t * self.nx: (t+1) * self.nx, t * self.nx: (t+1) * self.nx] = np.eye(self.nx)

            if t >=1:
                A[t * self.nx: (t+1) * self.nx, (t-1) * self.nx: t * self.nx] = - data.Fx.copy()

            B[t * self.nx: (t+1) * self.nx] = self.gap[t].copy()


        P[(self.problem.T-1)*self.nx:self.problem.T*self.nx, self.problem.T*self.nx-self.nx:self.problem.T*self.nx] = self.problem.terminalData.Lxx.copy()
        q[(self.problem.T-1)*self.nx:self.problem.T*self.nx] = self.problem.terminalData.Lx.copy()


        n = self.problem.T*(self.nx + self.nu)
        self.n_eq = self.problem.T*self.nx

        self.n_in = sum([cmodel.nc for cmodel in self.constraintModel])

        C = np.zeros((self.n_in, self.problem.T*(self.nx + self.nu)))
        l = np.zeros(self.n_in)
        u = np.zeros(self.n_in)

        nin_count = 0
        index_x = self.problem.T*self.nx
        for t, (cmodel, cdata) in enumerate(zip(self.constraintModel, self.constraintData)):
            if cmodel.nc == 0:
                continue
            l[nin_count: nin_count + cmodel.nc] = cmodel.lmin - cdata.c
            u[nin_count: nin_count + cmodel.nc] = cmodel.lmax - cdata.c
            if t > 0:
                C[nin_count: nin_count + cmodel.nc, (t-1)*self.nx: t*self.nx] = cdata.Cx
            if t < self.problem.T:
                C[nin_count: nin_count + cmodel.nc, index_x+t*self.nu: index_x+(t+1)*self.nu] = cdata.Cu
            nin_count += cmodel.nc


        # import pdb; pdb.set_trace()
        if self.method == "ProxQP":
            qp = proxsuite.proxqp.dense.QP(n, self.n_eq, self.n_in)
            qp.settings.eps_abs = 1e-5
            qp.init(P, q, A, B, C, l, u)      
            t1 = time.time()
            qp.solve()
            print("solve time = ", time.time()-t1)
            res = qp.results.x
            # print("n_iter = ", qp.results.info.iter)
            # print("n_iter_ext = ", qp.results.info.iter_ext)

        elif self.method == "OSQP":
            Aeq = sparse.csr_matrix(A)
            Aineq = sparse.csr_matrix(C)
            Aosqp = sparse.vstack([Aeq, Aineq])

            losqp = np.hstack([B, l])
            uosqp = np.hstack([B, u])

            P = sparse.csr_matrix(P)
            prob = osqp.OSQP()
            prob.setup(P, q, Aosqp, losqp, uosqp, warm_start=False, scaling=False,  max_iter = self.max_iters, \
                            adaptive_rho=True, verbose = self.verboseQP)     

            # t1 = time.time()
            tmp = prob.solve()
            res = tmp.x
            self.y_k = tmp.y
            self.QP_iter = tmp.info.iter
            # self.r_prim = tmp.primal_residual
            # print("solve time = ", time.time()-t1)

        elif self.method == "CustomOSQP" :
            Aeq = sparse.csr_matrix(A)
            Aineq = sparse.csr_matrix(C)
            self.Aosqp = sparse.vstack([Aeq, Aineq])
            
            self.losqp = np.hstack([B, l])
            self.uosqp = np.hstack([B, u])

            self.P = P
            self.q = np.array(q)
            
            self.x_k = np.zeros(self.n_vars)
            self.z_k = np.zeros(self.n_in + self.n_eq)
            self.y_k = np.zeros(self.n_in + self.n_eq)
            res = self.optimize_osqp(maxiters=self.max_iters)

        elif self.method == "FAdmmKKT":
            self.A_eq = sparse.csr_matrix(A.copy())
            self.A_in = sparse.csr_matrix(C.copy())
            self.b = B.copy()
            self.lboyd = l.copy()
            self.uboyd = u.copy()

            self.P = P.copy()
            self.q = np.array(q).copy()
            
            if self.initialize:
                self.xs_vec = np.array(self.xs).flatten()[self.nx:]
                self.us_vec = np.array(self.us).flatten()
                self.x_k = np.zeros_like(np.hstack((self.xs_vec, self.us_vec)))
                self.z_k = np.zeros(self.n_in)
                self.y_k = np.zeros(self.n_in)

                self.initialize = False

            res = self.optimize_boyd(maxiters=self.max_iters)

        self.dx[0] = np.zeros(self.nx)
        for t in range(self.problem.T):
            self.dx[t+1] = res[t * self.nx: (t+1) * self.nx] 
            index_u = self.problem.T*self.nx + t * self.nu
            self.du[t] = res[index_u:index_u+self.nu]

        self.x_grad_norm = np.linalg.norm(self.dx)/(self.problem.T+1)
        self.u_grad_norm = np.linalg.norm(self.du)/self.problem.T

    def acceptStep(self, alpha):
        
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.xs_try[t] = model.state.integrate(self.xs[t], alpha*self.dx[t])
            self.us_try[t] = self.us[t] + alpha*self.du[t]    
        self.xs_try[-1] = model.state.integrate(self.xs[-1], alpha*self.dx[-1]) ## terminal state update

        self.setCandidate(self.xs_try, self.us_try, False)

    def solve(self, init_xs=None, init_us=None, maxiter=1000, isFeasible=False, regInit=None):
        #___________________ Initialize ___________________#
        self.max_iters = maxiter
        if init_xs is None or len(init_xs) < 1:
            init_xs = [self.problem.x0.copy() for m in self.models()] 
        if init_us is None or len(init_us) < 1:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels] 

        init_xs[0][:] = self.problem.x0.copy() # Initial condition guess must be x0
        self.setCandidate(init_xs, init_us, False)
        self.computeDirectionFullQP()
        self.acceptStep(alpha = 1.0)
        # self.reset_params()
        
    def allocateData(self):    
        #
        self.xs_try = [np.zeros(m.state.nx) for m in self.models()] 
        self.xs_try[0][:] = self.problem.x0.copy()
        self.us_try = [np.zeros(m.nu) for m in self.problem.runningModels] 
        #
        self.dx = [np.zeros(m.state.ndx) for m  in self.models()]
        self.du = [np.zeros(m.nu) for m  in self.problem.runningModels] 

        self.constraintData = [cmodel.createData() for cmodel in self.constraintModel]
        self.dz_relaxed = [np.zeros(cmodel.nc) for cmodel in self.constraintModel]
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
