## Computes all the constraints at a particular time step
## Author : Armand Jordana
## Date : 12/04/2023

import numpy as np
from . abstract_model import ConstraintModelAbstact

class ConstraintModelStack(ConstraintModelAbstact):
    def __init__(self, constraintmodels, nx, nu):
        self.nc = sum([cmodel.nc for cmodel in constraintmodels])
        ConstraintModelAbstact.__init__(self, self.nc, nx, nu)
        self.lmin = np.concatenate([cmodel.lmin for cmodel in constraintmodels])
        self.lmax = np.concatenate([cmodel.lmax for cmodel in constraintmodels])
    
        self.cdatas = [cmodel.createData() for cmodel in constraintmodels]
        self.cmodels = constraintmodels

    def calc(self, cdata, data, x, u=None): 
        for (ci, di) in zip(self.cmodels, self.cdatas):
            ci.calc(di, data, x, u)
        cdata.c = np.concatenate([cdata.c for cdata in self.cdatas])

    def calcDiff(self, cdata, data, x, u=None):
        for (ci, di) in zip(self.cmodels, self.cdatas):
            ci.calcDiff(di, data, x, u)
        cdata.Cx = np.concatenate([cdata.Cx for cdata in self.cdatas])    
        cdata.Cu = np.concatenate([cdata.Cu for cdata in self.cdatas])    
    
    
    def compute_CxCu(self, dx, du):
        """returns Cx @ dx , Cu @ du"""
        z = np.zeros(self.nc)
        count = 0
        for (ci, di) in zip(self.cmodels, self.cdatas):
            z[count: count + ci.nc] = di.Cx @ dx + di.Cu @ du
            count += ci.nc
        return z

    def compute_transpose(self, vec):
        """returns Cx.T @ vec , Cu.T @ vec"""
        pass

    def compute_ADMM_hessian(self, rho_vec):
        """returns Cx.T @ rho_mat@ Cx, Cx.T @ rho_mat@ Cu, Cu.T @ rho_mat@ Cu ,"""
        pass