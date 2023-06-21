## This is the abstract model class to enforce constraints in the F-ADMM solver
## Author : Armand Jordana
## Date : 12/04/2023

import numpy as np
import pinocchio
import pinocchio as pin

try:
    from crocoddyl import ConstraintModelAbstract, NoConstraintModel
except:
    print("USING PYTHON ABSTRACT")

    class ConstraintModelAbstract():
        def __init__(self, state, nc, nu, lb, ub):
            self.state = state
            self.nc = nc
            self.nu = nu
            self.lb = lb
            self.ub = ub

        def createData(self):
            data = ConstraintData(self)
            return data
            
    class ConstraintData():
        def __init__(self, cmodel):
            self.c = np.zeros(cmodel.nc)
            self.Cx = np.zeros((cmodel.nc, cmodel.state.nx))
            self.Cu = np.zeros((cmodel.nc, cmodel.nu))


    class NoConstraintModel(ConstraintModelAbstract):
        def __init__(self, state, nu):
            ConstraintModelAbstract.__init__(self, state, 0, nu, np.zeros(0), np.zeros(0))

        def calc(self, cdata, data, x, u=None): 
            pass

        def calcDiff(self, cdata, data, x, u=None):
            pass
