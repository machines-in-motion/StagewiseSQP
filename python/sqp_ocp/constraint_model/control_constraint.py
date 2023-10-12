## This enforces control limits on the robot
## Author : Armand Jordana
## Date : 12/04/2023



CROCODDYL_VERSION = "fork_v1"

if CROCODDYL_VERSION == "fork_v1":
  from crocoddyl import ControlConstraintModel 
else:
  print("USING CONTROL PYTHON")
  import numpy as np
  from . abstract_model import ConstraintModelAbstract

  class ControlConstraintModel(ConstraintModelAbstract):
      def __init__(self, state, nu, lumin, lumax):
        ConstraintModelAbstract.__init__(self, state, nu, nu, lumin, lumax)
        self.Cx = np.zeros((len(lumin), state.nx))
        self.Cu = np.eye(nu)

      def calc(self, cdata, data, x, u=None): 
        cdata.c = u

      def calcDiff(self, cdata, data, x, u=None): 
        cdata.Cx = self.Cx 
        cdata.Cu = self.Cu

