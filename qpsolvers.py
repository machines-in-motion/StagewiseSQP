## This file contains methods to create matrices and solve using prox qp and OSQP for the 
## LQ sub problem 
## Author : Armand Jordana
## Date : 21/03/2022

import numpy as np
import osqp
import proxsuite
import time 
from scipy import sparse
import scipy.linalg as scl


class QPSolvers:

    def __init__(self, method):
        
        assert method == "ProxQP" or method=="sparceADMM"  or method=="OSQP"

        self.method = method

    def computeDirectionFullQP(self):
        self.calc(True)
        
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
            A[t * self.nx: (t+1) * self.nx, index_u:index_u+self.nu] = - data.Fu 
            A[t * self.nx: (t+1) * self.nx, t * self.nx: (t+1) * self.nx] = np.eye(self.nx)

            if t >=1:
                A[t * self.nx: (t+1) * self.nx, (t-1) * self.nx: t * self.nx] = - data.Fx

            B[t * self.nx: (t+1) * self.nx] = self.gap[t]


        P[self.problem.T*self.nx-self.nx:self.problem.T*self.nx, self.problem.T*self.nx-self.nx:self.problem.T*self.nx] = self.problem.terminalData.Lxx.copy()
        q[self.problem.T*self.nx-self.nx:self.problem.T*self.nx] = self.problem.terminalData.Lx.copy()


        n = self.problem.T*(self.nx + self.nu)
        n_eq = self.problem.T*self.nx
        n_in = self.problem.T*(self.nx + self.nu)


        n_in_x = sum([cmodel.ncx for cmodel in self.constraintModel[1:]])
        n_in_u = sum([cmodel.ncu for cmodel in self.constraintModel[:-1]])

        n_in = n_in_x + n_in_u
        C = np.zeros((n_in, n_in))
        l = np.zeros(n_in)
        u = np.zeros(n_in)

        nin_count = 0
        for t, cmodel in enumerate(self.constraintModel[1:]): 
            cx, _ =  cmodel.calc(self.xs[t+1])
            Cx, _ =  cmodel.calcDiff(self.xs[t], self.us[t])
            l[nin_count: nin_count + cmodel.ncx] = cmodel.lxmin - cx
            u[nin_count: nin_count + cmodel.ncx] = cmodel.lxmax - cx 
            C[nin_count: nin_count + cmodel.ncx, nin_count: nin_count + cmodel.ncx] = Cx
            nin_count += cmodel.ncx
         
        for t, cmodel in enumerate(self.constraintModel[:-1]): 
            _, cu =  cmodel.calc(self.xs[t], self.us[t])
            _, Cu =  cmodel.calcDiff(self.xs[t], self.us[t])
            l[nin_count: nin_count + cmodel.ncu] = self.constraintModel[t].lumin - cu
            u[nin_count: nin_count + cmodel.ncu] = self.constraintModel[t].lumax - cu
            C[nin_count: nin_count + cmodel.ncu, nin_count: nin_count + cmodel.ncu] = Cu
            nin_count += cmodel.ncu

        if self.method == "ProxQP":
            qp = proxsuite.proxqp.sparse.QP(n, n_eq, n_in)
            qp.init(P, q, A, B, C, l, u)      
            t1 = time.time()
            qp.solve()
            print("solve time = ", time.time()-t1)
            res = qp.results
            print("n_iter = ", qp.results.info.iter)
            print("n_iter_ext = ", qp.results.info.iter_ext)

        elif self.method == "OSQP":
            Aeq = sparse.csr_matrix(A)
            Aineq = sparse.eye(n_in)
            Aosqp = sparse.vstack([Aeq, Aineq])

            losqp = np.hstack([B, l])
            uosqp = np.hstack([B, u])

            P = sparse.csr_matrix(P)
            prob = osqp.OSQP()
            prob.setup(P, q, Aosqp, losqp, uosqp, warm_start=True, scaling=False)     

            t1 = time.time()
            res = prob.solve()
            print("solve time = ", time.time()-t1)
        # import pdb; pdb.set_trace()

        self.dx[0] = np.zeros(self.nx)
        for t in range(self.problem.T):
            self.dx[t+1] = res.x[t * self.nx: (t+1) * self.nx] 
            index_u = self.problem.T*self.nx + t * self.nu
            self.du[t] = res.x[index_u:index_u+self.nu]

        self.x_grad_norm = np.linalg.norm(self.dx)/(self.problem.T+1)
        self.u_grad_norm = np.linalg.norm(self.du)/self.problem.T
