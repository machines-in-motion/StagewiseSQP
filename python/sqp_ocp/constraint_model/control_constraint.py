## This enforces control limits on the robot
## Author : Armand Jordana
## Date : 12/04/2023


try:
  from crocoddyl import ControlConstraintModel 

except:
  print("USING CONTROL PYTHON")
  import numpy as np
  from . abstract_model import ConstraintModelAbstract

  class ControlConstraintModel(ConstraintModelAbstract):
      def __init__(self, state, nu, lumin, lumax):
        print("YES")

        ConstraintModelAbstract.__init__(self, state, nu, nu, lumin, lumax)
        print("YES")
        self.Cx = np.zeros((len(lumin), state.nx))
        self.Cu = np.eye(nu)

      def calc(self, cdata, x, u=None): 
        cdata.c = u

      def calcDiff(self, cdata, x, u=None): 
        cdata.Cx = self.Cx 
        cdata.Cu = self.Cu

