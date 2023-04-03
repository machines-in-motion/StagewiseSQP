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
        self.rho_min = 1e-1
        self.rho_max = 1e3
        self.alpha = 1.6

        self.eps_abs = 1e-3
        self.eps_rel = 1e-4

    def solve_linear_system(self):

        # A_block_leftcol = sparse.vstack([self.P + self.sigma * np.eye(self.n_vars), self.Aosqp])
        # A_block_rightcol = sparse.vstack([self.Aosqp.T, (-1/self.rho)*np.eye(self.n_in + self.n_eq)])
        # A_block = sparse.hstack([A_block_leftcol, A_block_rightcol])

        # b_block = np.hstack((self.sigma*self.x_k - self.q, self.z_k - np.divide(self.y_k, self.rho)))
        # xv_k_1 = spsolve(A_block, b_block)
        # self.xtilde_k_1, self.v_k_1 = xv_k_1[:self.n_vars], xv_k_1[self.n_vars:]
        # self.ztilde_k_1 = self.z_k + np.divide(self.v_k_1 - self.y_k, self.rho)

        rho_mat = self.rho*np.eye(len(self.rho))
        A_block = self.P + self.sigma * np.eye(self.n_vars) + (self.Aosqp.T@rho_mat@self.Aosqp)
        b_block = self.sigma*self.x_k - self.q + self.Aosqp.T@(np.multiply(self.rho,self.z_k) - self.y_k)

        self.xtilde_k_1 = spsolve(A_block, b_block)
        self.ztilde_k_1 = self.Aosqp@self.xtilde_k_1

    def update_lagrangian_params(self):

        self.x_k_1 = self.alpha * self.xtilde_k_1 + (1-self.alpha)*self.x_k
        self.z_k_1 = np.clip(self.alpha*self.ztilde_k_1 + (1 - self.alpha)*self.z_k + np.divide(self.y_k,self.rho), self.losqp, self.uosqp)
        self.y_k_1 = self.y_k + np.multiply(self.rho, (self.alpha*self.ztilde_k_1 + (1 - self.alpha)*self.z_k - self.z_k_1))

        self.x_k, self.z_k, self.y_k = self.x_k_1, self.z_k_1, self.y_k_1

        self.r_prim = max(abs(self.Aosqp @ self.x_k - self.z_k))
        self.r_dual = max(abs(self.P @ self.x_k + self.q + self.Aosqp.T @ self.y_k))

        self.eps_rel_prim = max(abs(np.hstack((self.Aosqp @ self.x_k, self.z_k))))
        tmp = max(abs(self.q))
        tmp2 = max(abs(self.Aosqp.T @ self.y_k))
        tmp3 = max(abs(self.P @ self.x_k))
        self.eps_rel_dual = max(tmp, tmp2)
        self.eps_rel_dual = max(self.eps_rel_dual, tmp3)

    def update_rho(self):
        
        scale = np.sqrt(self.r_prim * self.eps_rel_dual/(self.r_dual * self.eps_rel_prim))
        self.rho_op *= scale
        for i in range(len(self.losqp)):
            if self.losqp[i] == -np.inf and self.uosqp[i] == np.inf:
                self.rho[i] = self.rho_min
            elif abs(self.losqp[i] - self.uosqp[i]) < 1e-4:
                self.rho[i] = 1e3*self.rho_op
            elif self.losqp[i] != self.uosqp[i]:
                self.rho[i] = self.rho_op
            else:
                assert False # safety check
        self.rho_estimate = min(max(self.rho_op, self.rho_min), self.rho_max) 

    def set_rho(self):

        self.rho = np.zeros(self.n_eq + self.n_in)
        for i in range(len(self.losqp)):
            if self.losqp[i] == -np.inf and self.uosqp[i] == np.inf:
                self.rho[i] = self.rho_min
            elif self.losqp[i] == self.uosqp[i]:
                self.rho[i] = 1e3 * self.rho_op
            elif self.losqp[i] != self.uosqp[i]:
                self.rho[i] = self.rho_op

    def optimize_osqp(self, maxiters = 1000):

        self.set_rho()
        for i in range(maxiters):
            self.solve_linear_system()
            self.update_lagrangian_params()
            self.update_rho()

            eps_prim = self.eps_abs + self.eps_rel * self.eps_rel_prim
            eps_dual = self.eps_abs + self.eps_rel * self.eps_rel_dual

            # print(self.r_prim, self.r_dual, self.eps_rel_dual)

            if self.r_prim < eps_prim and self.r_dual < eps_dual:
                print("terminated ... \n")
                print("Iters", i, "res-primal", eps_prim, "res-dual", eps_dual, "optimal rho", self.rho_estimate)
                break
            # print("Iters", i, "res-primal", eps_prim, "res-dual", eps_dual, "optimal rho", self.rho_estimate)

        return self.x_k
            