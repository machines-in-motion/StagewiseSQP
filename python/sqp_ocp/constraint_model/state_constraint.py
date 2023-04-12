## This enforces kinematic constrains on the states of the robot
## Author : Armand Jordana
## Date : 12/04/2023

import numpy as np
from . abstract_model import ConstraintModelAbstact


class StateConstraintModel(ConstraintModelAbstact):
    def __init__(self, lxmin, lxmax, nc, nx, nu):
      ConstraintModelAbstact.__init__(self, nc, nx, nu)
      self.lmin = lxmin
      self.lmax = lxmax

      self.Cx = np.eye(len(lxmin))
      self.Cu = np.zeros((len(lxmin), nu))
      self.nc = len(lxmin)

    def calc(self, cdata, data, x, u=None): 
      cdata.c = x

    def calcDiff(self, cdata, data, x, u=None): 
      cdata.Cx = self.Cx 
      cdata.Cu = self.Cu
