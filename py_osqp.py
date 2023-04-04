## Custom osqp implementation
## Author : Avadesh Meduri
## Date : 31/03/2023

import numpy as np
from scipy import sparse
import scipy.linalg as scl
from scipy.sparse.linalg import spsolve
np.set_printoptions(precision = 2) 
# np.format_float_scientific(self.r_prim, exp_digits=2, precision =2)

pp = lambda s : np.format_float_scientific(s, exp_digits=2, precision =2)

class CustomOSQP():

    def __init__(self):

        self.sigma = 1e-6
        self.rho = 1e-1
        self.rho_min = 1e-6
        self.rho_max = 1e6
        self.alpha = 1.6

        self.eps_abs = 1e-3
        self.eps_rel = 1e-3
        self.adaptive_rho_tolerance = 5
        self.rho_update_interval = 25

    def solve_linear_system(self):

        A_block_leftcol = sparse.vstack([self.P + self.sigma * np.eye(self.n_vars), self.Aosqp])
        A_block_rightcol = sparse.vstack([self.Aosqp.T, (-1/self.rho_vec)*np.eye(self.n_in + self.n_eq)])
        A_block = sparse.hstack([A_block_leftcol, A_block_rightcol])
        A_block = sparse.csr_matrix(A_block)

        b_block = np.hstack((self.sigma*self.x_k - self.q, self.z_k - np.divide(self.y_k, self.rho_vec)))
        
        xv_k_1 = spsolve(A_block, b_block)
        self.xtilde_k_1, self.v_k_1 = xv_k_1[:self.n_vars], xv_k_1[self.n_vars:]
        self.ztilde_k_1 = self.z_k + np.divide(self.v_k_1 - self.y_k, self.rho_vec)
        # self.ztilde_k_1 = self.v_k_1.copy()
        # rho_mat = self.rho_vec*np.eye(len(self.rho_vec))
        # A_block = self.P + self.sigma * np.eye(self.n_vars) + (self.Aosqp.T@rho_mat@self.Aosqp)
        # b_block = self.sigma*self.x_k - self.q + self.Aosqp.T@(np.multiply(self.rho_vec,self.z_k) - self.y_k)

        # self.xtilde_k_1 = spsolve(A_block, b_block)
        # self.ztilde_k_1 = self.Aosqp@self.xtilde_k_1

    def update_lagrangian_params(self):

        self.x_k_1 = self.alpha * self.xtilde_k_1 + (1-self.alpha)*self.x_k
        self.z_k_1 = np.clip(self.alpha*self.ztilde_k_1 + (1 - self.alpha)*self.z_k + np.divide(self.y_k,self.rho_vec), self.losqp, self.uosqp)
        self.y_k_1 = self.y_k + np.multiply(self.rho_vec, (self.alpha*self.ztilde_k_1 + (1 - self.alpha)*self.z_k - self.z_k_1))

        self.x_k, self.z_k, self.y_k = self.x_k_1, self.z_k_1, self.y_k_1

        self.r_prim = max(abs(self.Aosqp @ self.x_k - self.z_k))
        self.r_dual = max(abs(self.P @ self.x_k + self.q + self.Aosqp.T @ self.y_k))

        self.eps_rel_prim = max(abs(np.hstack((self.Aosqp @ self.x_k, self.z_k))))
        tmp = max(abs(self.q))
        tmp2 = max(abs(self.Aosqp.T @ self.y_k))
        tmp3 = max(abs(self.P @ self.x_k))
        self.eps_rel_dual = max(tmp, tmp2)
        self.eps_rel_dual = max(self.eps_rel_dual, tmp3)

    def update_rho(self, iter):
        
        scale = np.sqrt(self.r_prim * self.eps_rel_dual/(self.r_dual * self.eps_rel_prim))
        self.rho_estimate = scale * self.rho
        self.rho_estimate = min(max(self.rho_estimate, self.rho_min), self.rho_max) 
    
        if (iter) % self.rho_update_interval == 0 and iter > 1:
            if self.rho_estimate > self.rho * self.adaptive_rho_tolerance or\
            self.rho_estimate < self.rho / self.adaptive_rho_tolerance :
                self.rho = self.rho_estimate
                for i in range(len(self.losqp)):
                    if self.losqp[i] == -np.inf and self.uosqp[i] == np.inf:
                        self.rho_vec[i] = self.rho_min
                    elif abs(self.losqp[i] - self.uosqp[i]) < 1e-4:
                        self.rho_vec[i] = 1e3*self.rho
                    elif self.losqp[i] != self.uosqp[i]:
                        self.rho_vec[i] = self.rho
                    else:
                        assert False # safety check

    def set_rho(self):

        self.rho_vec = np.zeros(self.n_eq + self.n_in)
        self.rho = min(max(self.rho, self.rho_min), self.rho_max) 
        for i in range(len(self.losqp)):
            if self.losqp[i] == -np.inf and self.uosqp[i] == np.inf:
                self.rho_vec[i] = self.rho_min
            elif self.losqp[i] == self.uosqp[i]:
                self.rho_vec[i] = 1e3 * self.rho
            elif self.losqp[i] != self.uosqp[i]:
                self.rho_vec[i] = self.rho
        self.rho_estimate = 0

    def optimize_osqp(self, maxiters = 1000):

        self.set_rho()
        converged = False
        for iter in range(1, maxiters+1):
            self.solve_linear_system()
            self.update_lagrangian_params()
            self.update_rho(iter)

            eps_prim = self.eps_abs + self.eps_rel * self.eps_rel_prim
            eps_dual = self.eps_abs + self.eps_rel * self.eps_rel_dual

            if (iter) % self.rho_update_interval == 0 and iter > 1:
                if self.r_prim < eps_prim and self.r_dual < eps_dual:
                    print("terminated ... \n")
                    print("Iters", iter, "res-primal", pp(self.r_prim), "res-dual", pp(self.r_dual)\
                , "optimal rho estimate", pp(self.rho_estimate), "rho", pp(self.rho)) 
                    converged = True               
                    break
        if not converged:
            print("Iters", iter, "res-primal", pp(self.r_prim), "res-dual", pp(self.r_dual)\
                , "optimal rho estimate", pp(self.rho_estimate), "rho", pp(self.rho))
        return self.x_k
            