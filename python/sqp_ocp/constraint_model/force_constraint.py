## This file contains various constraints on the forces (friction, unilaterality ...)
## Author : Armand Jordana
## Date : 12/04/2023

import numpy as np
from . abstract_model import ConstraintModelAbstract
import crocoddyl


class Force6DConstraintModel(ConstraintModelAbstract):
    def __init__(self, state, nu, Fmin, Fmax):
        ConstraintModelAbstract.__init__(self, state, 6, nu, Fmin, Fmax)

    def calc(self, cdata, data, x, u=None): 
        cdata.c = data.differential.pinocchio.lambda_c

    def calcDiff(self, cdata, data, x, u=None):
        cdata.Cx = data.differential.df_dx
        cdata.Cu = data.differential.df_du

class Force3DConstraintModel(crocoddyl.ConstraintModelAbstract):
    def __init__(self, Fmin, Fmax, nc, nx, nu):
        ConstraintModelAbstract.__init__(self, nc, nx, nu)
        self.lmin = Fmin
        self.lmax = Fmax

        self.nc = 3

    def calc(self, cdata, data, x, u=None): 
        cdata.c = data.differential.pinocchio.lambda_c[:3]

    def calcDiff(self, cdata, data, x, u=None):
        cdata.Cx = data.differential.df_dx[:3]
        cdata.Cu = data.differential.df_du[:3]


class LocalCone(ConstraintModelAbstract):
    def __init__(self, state, nu, mu):
        ConstraintModelAbstract.__init__(self, state, 1, nu, np.array([0.]), np.array([np.inf]))
        self.mu = mu
        self.dcone_df = np.zeros((1, 3))


    def calc(self, cdata, data, x, u=None): 
        F = data.differential.pinocchio.lambda_c[:3]
        cdata.c = - self.mu * F[2] - np.sqrt(F[0]**2 + F[1]**2)

    def calcDiff(self, cdata, data, x, u=None):
        F = data.differential.pinocchio.lambda_c[:3]
        Fx = data.differential.df_dx[:3]
        Fu = data.differential.df_du[:3]

        self.dcone_df[0, 0] = - F[0] / np.sqrt(F[0]**2 + F[1]**2)
        self.dcone_df[0, 1] = - F[1] / np.sqrt(F[0]**2 + F[1]**2)
        self.dcone_df[0, 2] = - self.mu
        cdata.Cx = self.dcone_df @ data.differential.df_dx[:3]
        cdata.Cu = self.dcone_df @ data.differential.df_du[:3]