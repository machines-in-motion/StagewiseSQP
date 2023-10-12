## Computes all the constraints at a particular time step
## Author : Armand Jordana
## Date : 12/04/2023

CROCODDYL_VERSION = "fork_v1"

if CROCODDYL_VERSION == "fork_v1":
    from crocoddyl import ConstraintStack as ConstraintModelStack
else:
    import numpy as np

    from . abstract_model import ConstraintModelAbstract

    class ConstraintModelStack(ConstraintModelAbstract):
        def __init__(self, constraintmodels, state, nc, nu, name):

            self.cmodels = constraintmodels
            self.cdatas = [cmodel.createData() for cmodel in constraintmodels]

            lmin = np.concatenate([cmodel.lb for cmodel in constraintmodels])
            lmax = np.concatenate([cmodel.ub for cmodel in constraintmodels])

            ConstraintModelAbstract.__init__(self, state, nc, nu, lmin, lmax)


        def calc(self, cdata, data, x, u=None):
            count = 0 
            for (ci, di) in zip(self.cmodels, self.cdatas):
                ci.calc(di, data, x, u)
                cdata.c[count:count + ci.nc] = di.c
                count += ci.nc

        def calcDiff(self, cdata, data, x, u=None):
            count = 0
            for (ci, di) in zip(self.cmodels, self.cdatas):
                ci.calcDiff(di, data, x, u)
                cdata.Cx[count:count + ci.nc] = di.Cx     
                cdata.Cu[count:count + ci.nc] = di.Cu     
        
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



