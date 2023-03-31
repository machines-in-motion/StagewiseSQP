## Custom osqp implementation
## Author : Avadesh Meduri
## Date : 31/03/2023

import numpy as np
from scipy import sparse
import scipy.linalg as scl
from scipy.sparse.linalg import spsolve

class CustomOSQP():

    def __init__(self):

        self.sigma = 1e-6
        self.rho_op = 1e-1
        self.alpha = 1.6

        self.eps_abs = 1e-3
        self.eps_rel = 1e-3

    def create_matrices(self):

        self.calc(True)
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
            A[t * self.nx: (t+1) * self.nx, index_u:index_u+self.nu] = - data.Fu 
            A[t * self.nx: (t+1) * self.nx, t * self.nx: (t+1) * self.nx] = np.eye(self.nx)

            if t >=1:
                A[t * self.nx: (t+1) * self.nx, (t-1) * self.nx: t * self.nx] = - data.Fx

            B[t * self.nx: (t+1) * self.nx] = self.gap[t]


        P[self.problem.T*self.nx-self.nx:self.problem.T*self.nx, self.problem.T*self.nx-self.nx:self.problem.T*self.nx] = self.problem.terminalData.Lxx.copy()
        q[self.problem.T*self.nx-self.nx:self.problem.T*self.nx] = self.problem.terminalData.Lx.copy()


        n = self.problem.T*(self.nx + self.nu)
        self.n_eq = self.problem.T*self.nx
        self.n_in = self.problem.T*(self.nx + self.nu)

        C = np.eye(self.n_in)
        l = np.zeros(self.n_in)
        u = np.zeros(self.n_in)

        for t in range(self.problem.T): 
            l[t * self.nx: (t+1) * self.nx] = self.lxmin[t+1] - self.xs[t+1]
            u[t * self.nx: (t+1) * self.nx] = self.lxmax[t+1] - self.xs[t+1] 
            index_u = self.problem.T*self.nx + t * self.nu
            l[index_u: index_u + self.nu] = self.lumin[t] - self.us[t]
            u[index_u: index_u + self.nu] = self.lumax[t] - self.us[t]

        Aeq = sparse.csr_matrix(A)
        Aineq = sparse.eye(self.n_in)
        self.Aosqp = sparse.vstack([Aeq, Aineq])

        self.losqp = np.hstack([B, l])
        self.uosqp = np.hstack([B, u])

        self.P = P
        self.q_arr = np.array(q)
        
        self.xs_vec = np.array(self.xs).flatten()[self.nx:]
        self.us_vec = np.array(self.us).flatten()
        self.xz_vec = np.array(self.xz).flatten()[self.nx:]
        self.uz_vec = np.array(self.uz).flatten()
        self.xy_vec = np.array(self.xy).flatten()[self.nx:]
        self.uy_vec = np.array(self.uy).flatten()
        self.x_k = np.hstack((self.xs_vec, self.us_vec))
        # self.z_k = np.hstack((self.xz_vec, self.uz_vec))
        # self.y_k = np.hstack((self.xy_vec, self.uy_vec))
        self.z_k = np.zeros(self.n_in + self.n_eq)
        self.y_k = np.zeros(self.n_in + self.n_eq)

        ## initializing rho
        self.rho = self.rho_op * np.ones(self.n_eq + self.n_in)
        self.rho[:self.n_eq] *= 1e3
        self.rho_estimate = self.rho_op

    def solve_linear_system(self):

        A_block_leftcol = sparse.vstack([self.P + self.sigma * np.eye(self.n_vars), self.Aosqp])
        A_block_rightcol = sparse.vstack([self.Aosqp.T, (-1/self.rho)*np.eye(self.n_in + self.n_eq)])
        A_block = sparse.hstack([A_block_leftcol, A_block_rightcol])

        b_block = np.hstack((self.sigma*self.x_k - self.q_arr, self.z_k - np.divide(self.y_k, self.rho)))

        xv_k_1 = spsolve(A_block, b_block)
        self.xtilde_k_1, self.v_k_1 = xv_k_1[:self.n_vars], xv_k_1[self.n_vars:]

    def update_lagrangian_params(self):

        self.ztilde_k_1 = self.z_k + np.divide(self.v_k_1 - self.y_k, self.rho)
        self.x_k_1 = self.alpha * self.xtilde_k_1 + (1-self.alpha)*self.x_k
        self.z_k_1 = np.clip(self.alpha*self.ztilde_k_1 + (1 - self.alpha)*self.z_k + np.divide(self.y_k,self.rho), self.losqp, self.uosqp)
        self.y_k_1 = self.y_k + np.multiply(self.rho, (self.alpha*self.ztilde_k_1 + (1 - self.alpha)*self.z_k - self.z_k_1))

        self.x_k, self.z_k, self.y_k = self.x_k_1, self.z_k_1, self.y_k_1

        self.r_prim = max(abs(self.Aosqp @ self.x_k - self.z_k))
        self.r_dual = max(abs(self.P @ self.x_k + self.q_arr + self.Aosqp.T @ self.y_k))

        self.eps_rel_prim = max(abs(np.hstack((self.Aosqp @ self.x_k, self.z_k))))
        self.eps_rel_dual = max(abs(np.hstack((self.P @ self.x_k, self.Aosqp.T @ self.y_k, self.q_arr))))

    def update_rho(self):
        
        scale = np.sqrt(self.r_prim * self.eps_rel_dual/(self.r_dual * self.eps_rel_prim))
        self.rho *= scale
        self.rho_estimate *= scale 

    def optimize_osqp(self, maxiters = 1000):

        self.create_matrices()
        for i in range(maxiters):
            self.solve_linear_system()
            self.update_lagrangian_params()
            self.update_rho()
            eps_prim = self.eps_abs + self.eps_rel * self.eps_rel_prim
            eps_dual = self.eps_abs + self.eps_rel * self.eps_rel_dual
            if self.r_prim < eps_prim and self.r_dual < eps_dual:
                print("terminated ... \n")
                print("Iters", i, "res-primal", eps_prim, "res-dual", eps_dual, "optimal rho", self.rho_estimate)
                break

        self.dx[0] = np.zeros(self.nx)
        for t in range(self.problem.T):
            self.dx[t+1] = self.x_k[t * self.nx: (t+1) * self.nx] 
            index_u = self.problem.T*self.nx + t * self.nu
            self.du[t] = self.x_k[index_u:index_u+self.nu]
            
        self.x_grad_norm = np.linalg.norm(self.dx)/(self.problem.T+1)
        self.u_grad_norm = np.linalg.norm(self.du)/self.problem.T