## This enforces control limits on the robot
## Author : Armand Jordana
## Date : 12/04/2023

import numpy as np
from . abstract_model import ConstraintModelAbstact

class ControlConstraintModel(ConstraintModelAbstact):
    def __init__(self, lumin, lumax, nc, nx, nu):
      ConstraintModelAbstact.__init__(self, nc, nx, nu)
      self.lmin = lumin
      self.lmax = lumax

      self.Cx = np.zeros((len(lumin), nx))
      self.Cu = np.eye(nu)
      self.nc = len(lumin)

    def calc(self, cdata, data, x, u=None): 
      cdata.c = u

    def calcDiff(self, cdata, data, x, u=None): 
      cdata.Cx = self.Cx 
      cdata.Cu = self.Cu

