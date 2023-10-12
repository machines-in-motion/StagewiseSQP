## This enforces kinematic constrains on the states of the robot
## Author : Armand Jordana
## Date : 12/04/2023



CROCODDYL_VERSION = "fork_v1"

if CROCODDYL_VERSION == "fork_v1":
   from crocoddyl import StateConstraintModel
else:
  print("USING PYTHON CONSTRAINT")
  import numpy as np
  from . abstract_model import ConstraintModelAbstract


  class StateConstraintModel(ConstraintModelAbstract):
      def __init__(self, state, nu, lxmin, lxmax, name):
        ConstraintModelAbstract.__init__(self, state, state.nx, nu, lxmin, lxmax)

        self.Cx = np.eye(len(lxmin))
        self.Cu = np.zeros((len(lxmin), nu))

      def calc(self, cdata, data, x, u=None): 
        cdata.c = x

      def calcDiff(self, cdata, data, x, u=None): 
        cdata.Cx = self.Cx 
        cdata.Cu = self.Cu
