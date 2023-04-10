## This file contains methods to create matrices and solve using prox qp and OSQP for the 
## LQ sub problem 
## Author : Armand Jordana
## Date : 21/03/2022

import numpy as np
import osqp
import proxsuite
import time 
from scipy import sparse
from py_osqp import CustomOSQP
from py_boyd import BoydADMM

class QPSolvers(CustomOSQP, BoydADMM):

    def __init__(self, method):
        
        assert method == "ProxQP" or method=="sparceADMM"  or method=="OSQP"\
              or method=="CustomOSQP" or method =="Boyd" 
        if method == "CustomOSQP":
            CustomOSQP.__init__(self)
        if method == "Boyd":
            BoydADMM.__init__(self)
        self.method = method

    def computeDirectionFullQP(self, maxit = 5000):
        self.n_vars  = self.problem.T*(self.nx + self.nu)

        P = np.zeros((self.problem.T*(self.nx + self.nu), self.problem.T*(self.nx + self.nu)))
        q = np.zeros(self.problem.T*(self.nx + self.nu))
        
        Asize = self.problem.T*(self.nx + self.nu)
        A = np.zeros((self.problem.T*self.nx, Asize))
        B = np.zeros(self.problem.T*self.nx)

        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            if t>=1:
                index_x = (t-1) * self.nx
                P[index_x:index_x+self.nx, index_x:index_x+self.nx] = data.Lxx.copy()
                q[index_x:index_x+self.nx] = data.Lx.copy()

            index_u = self.problem.T*self.nx + t * self.nu
            P[index_u:index_u+self.nu, index_u:index_u+self.nu] = data.Luu.copy()
            q[index_u:index_u+self.nu] = data.Lu.copy()
            
            index_u = self.problem.T*self.nx + t * self.nu
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
            prob.setup(P, q, Aosqp, losqp, uosqp, warm_start=False, scaling=False,  max_iter = maxit, \
                            adaptive_rho=True, verbose = True)     

            t1 = time.time()
            tmp = prob.solve()
            res = tmp.x
            self.y_k = tmp.y
            # self.r_prim = tmp.primal_residual
            print("solve time = ", time.time()-t1)
        # import pdb; pdb.set_trace()

        elif self.method == "CustomOSQP" :
            Aeq = sparse.csr_matrix(A)
            Aineq = sparse.csr_matrix(C)
            self.Aosqp = sparse.vstack([Aeq, Aineq])
            
            self.losqp = np.hstack([B, l])
            self.uosqp = np.hstack([B, u])

            self.P = P
            self.q = np.array(q)
            
            self.xs_vec = np.array(self.xs).flatten()[self.nx:]
            self.us_vec = np.array(self.us).flatten()
            self.xz_vec = np.array(self.xz).flatten()[self.nx:]
            self.uz_vec = np.array(self.uz).flatten()
            self.xy_vec = np.array(self.xy).flatten()[self.nx:]
            self.uy_vec = np.array(self.uy).flatten()
            self.x_k = np.hstack((self.xs_vec, self.us_vec))

            self.z_k = np.zeros(self.n_in + self.n_eq)
            self.y_k = np.zeros(self.n_in + self.n_eq)

            res = self.optimize_osqp(maxiters=maxit)

        elif self.method == "Boyd":
            self.A_eq = sparse.csr_matrix(A.copy())
            self.A_in = sparse.csr_matrix(C.copy())
            self.b = B.copy()
            self.lboyd = l.copy()
            self.uboyd = u.copy()

            self.P = P.copy()
            self.q = np.array(q).copy()
            self.xs_vec = np.array(self.xs).flatten()[self.nx:]
            self.us_vec = np.array(self.us).flatten()
            self.x_k = np.zeros_like(np.hstack((self.xs_vec, self.us_vec)))

            self.z_k = np.zeros(self.n_in)
            self.y_k = np.zeros(self.n_in)

            res = self.optimize_boyd(maxiters=maxit)

        self.dx[0] = np.zeros(self.nx)
        for t in range(self.problem.T):
            self.dx[t+1] = res[t * self.nx: (t+1) * self.nx] 
            index_u = self.problem.T*self.nx + t * self.nu
            self.du[t] = res[index_u:index_u+self.nu]

        self.x_grad_norm = np.linalg.norm(self.dx)/(self.problem.T+1)
        self.u_grad_norm = np.linalg.norm(self.du)/self.problem.T
