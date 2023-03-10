## This is the implementation of the constrained LQR
## Author : Avadesh Meduri and Armand Jordana
## Date : 8/03/2023

import numpy as np
from numpy import linalg
import proxsuite
import crocoddyl
from crocoddyl import SolverAbstract
from scipy import sparse
import osqp
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


class CLQR(SolverAbstract):
    def __init__(self, shootingProblem):
        SolverAbstract.__init__(self, shootingProblem)
        
        self.rho_op = 1e1

        self.allocateData()

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
        
        # self.gap_norm = np.linalg.norm(self.gap, 1)
        self.gap_norm = sum(np.linalg.norm(self.gap, 1, axis = 1))

        self.cost += self.problem.terminalData.cost 

    def computeDirection(self,iters):
        self.calc(True)
        for i in range(iters):
            if i == 0:
                self.rho = 0.0 # This is the proxQP trick
            else:
                self.rho = self.rho_op
            self.backwardPass()  
            self.computeUpdates()
            self.update_lagrangian_parameters()



    def computeDirectionOSQP(self):
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

        C = np.eye(n_in)
        l = np.zeros(n_in)
        u = np.zeros(n_in)

        for t in range(self.problem.T): 
            l[t * self.nx: (t+1) * self.nx] = self.lxmin[t+1]
            u[t * self.nx: (t+1) * self.nx] = self.lxmax[t+1]
            index_u = self.problem.T*self.nx + t * self.nu
            l[index_u: index_u + self.nu] = self.lumin[t]
            u[index_u: index_u + self.nu] = self.lumax[t]

        # solve it
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.init(P, q, A, B, C, l, u)        
        qp.solve()
        # print an optimal solution
        # print("optimal x: {}".format(qp.results.x))

        # A = sparse.csr_matrix(A)
        # P = sparse.csr_matrix(P)
        # q = sparse.csr_matrix(q)
        # l = sparse.csr_matrix(B)
        # u = sparse.csr_matrix(B)
        # prob = osqp.OSQP()
        # prob.setup(P, q, A, l, u, warm_start=True)
        # res = prob.solve()
        self.dx[0] = np.zeros(self.nx)
        for t in range(self.problem.T):
            self.dx[t+1] = qp.results.x[t * self.nx: (t+1) * self.nx] 
            index_u = self.problem.T*self.nx + t * self.nu
            self.du[t] = qp.results.x[index_u:index_u+self.nu]

        self.x_grad_norm = np.linalg.norm(self.dx)/(self.problem.T+1)
        self.u_grad_norm = np.linalg.norm(self.du)/self.problem.T


    def update_lagrangian_parameters(self):
        ## hard coding clipping now

        norm_r = 0
        norm_dz = 0

        for t in range(self.problem.T):
            xz_old = self.xz[t]
            uz_old = self.uz[t]

            self.xz[t] = np.clip(self.dx[t] + self.xy[t], self.lxmin[t], self.lxmax[t])
            self.uz[t] = np.clip(self.du[t] + self.uy[t], self.lumin[t], self.lumax[t])
            self.xy[t] += self.dx[t] - self.xz[t] 
            self.uy[t] += self.du[t] - self.uz[t] 

            norm_dz += np.linalg.norm(self.xz[t] - xz_old) + np.linalg.norm(self.uz[t] - uz_old) 

            norm_r += np.linalg.norm(self.xz[t] - self.dx[t])
            norm_r += np.linalg.norm(self.uz[t] - self.du[t])

        xz_old = self.xz[-1]
    
        self.xz[-1] = np.clip(self.dx[-1] + self.xy[-1], self.lxmin[-1], self.lxmax[-1])
        norm_dz += np.linalg.norm(self.xz[-1] - xz_old)
        self.xy[-1] += self.dx[-1] - self.xz[-1] 
        norm_r += np.linalg.norm(self.xz[-1] - self.dx[-1])

        print(norm_r, norm_dz )

    def computeUpdates(self): 
        """ computes step updates dx and du """
        self.expected_decrease = 0
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
                # here we compute the direction 
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
        self.S[-1][:,:] = self.problem.terminalData.Lxx + self.rho*np.eye(self.problem.terminalModel.state.nx)
        self.s[-1][:] = self.problem.terminalData.Lx + self.rho*(self.xy[-1] - self.xz[-1])[:]
        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels,self.problem.runningDatas)):

            r = data.Lu + self.rho*(self.uy[t] - self.uz[t])[:]
            q = data.Lx + self.rho*(self.xy[t] - self.xz[t])[:]
            R = data.Luu + self.rho*np.eye(model.nu)
            Q = data.Lxx + self.rho*np.eye(model.state.nx)
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
                    print("increasing H")
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
        # self.computeDirection(200)
        self.computeDirectionOSQP()
        self.acceptStep(alpha = 1.0)
        
        print("Total cost", self.cost, "gap norms", self.gap_norm, "dx norm", self.x_grad_norm, "du norm", self.u_grad_norm)
        self.calc(True)
        print("Total cost", self.cost, "gap norms", self.gap_norm, "dx norm", self.x_grad_norm, "du norm", self.u_grad_norm)



    def allocateData(self):
        self.xs_try = [np.zeros(m.state.nx) for m in self.models()] 
        self.xs_try[0][:] = self.problem.x0.copy()
        self.us_try = [np.zeros(m.nu) for m in self.problem.runningModels] 
        # 
        self.dx = [np.zeros(m.state.ndx) for m  in self.models()]
        self.du = [np.zeros(m.nu) for m  in self.problem.runningModels] 
        # 
        self.xz = [np.zeros(m.state.nx) for m  in self.models()]
        self.uz = [np.zeros(m.nu) for m  in self.problem.runningModels] 
        self.xy = [np.zeros(m.state.nx) for m  in self.models()]
        self.uy = [np.zeros(m.nu) for m  in self.problem.runningModels] 
        #
        cl = np.inf
        tmp = np.array([0.2, 0.1, np.inf, np.inf])
        self.lxmin = [-cl*np.ones(m.state.nx) for m  in self.models()]
        self.lxmax = [tmp for m  in self.models()]
        self.lumin = [-cl*np.ones(m.nu) for m  in self.problem.runningModels] 
        self.lumax = [cl*np.ones(m.nu) for m  in self.problem.runningModels] 
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

        self.nx = self.problem.terminalModel.state.nx 
        self.nu = self.problem.runningModels[0].nu

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