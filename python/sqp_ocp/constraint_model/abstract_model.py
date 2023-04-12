## This is the abstract model class to enforce constraints in the F-ADMM solver
## Author : Armand Jordana
## Date : 12/04/2023

import numpy as np
import pinocchio
import pinocchio as pin


class ConstraintModelAbstact():
    def __init__(self, nc, nx, nu):
        self.nc = nc
        self.nx = nx
        self.nu = nu

    def createData(self):
        data = ConstraintData(self)
        return data
        

class ConstraintData():
    def __init__(self, cmodel):
        self.c = np.zeros(cmodel.nc)
        self.Cx = np.zeros((cmodel.nc, cmodel.nx))
        self.Cu = np.zeros((cmodel.nc, cmodel.nu))



class NoConstraint(ConstraintModelAbstact):
    def __init__(self, nx, nu):
        ConstraintModelAbstact.__init__(self, 0, nx, nu)
        self.nc = 0
        self.Cx = np.zeros((self.nc, nx))
        self.Cu = np.zeros((self.nc, nu))
        self.lmin = np.array([])
        self.lmax = np.array([])

    def calc(self, cdata, data, x, u=None): 
        cdata.c = np.array([])

    def calcDiff(self, cdata, data, x, u=None):
        cdata.Cx = self.Cx
        cdata.Cu = self.Cu

