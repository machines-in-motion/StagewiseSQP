import numpy as np
from pin_utils import get_p
import pinocchio as pin

class FullConstraintModel():
    def __init__(self, lxmin, lxmax, lumin, lumax):
      self.lxmin = lxmin
      self.lxmax = lxmax
      self.lumin = lumin
      self.lumax = lumax
      self.Cx = np.eye(len(lxmin))
      self.Cu = np.eye(len(lumin))

      self.ncx = len(lxmin)
      self.ncu = len(lumin)

    def calc(self, x, u=None): 
      return x, u

    def calcDiff(self, x, u=None): 
      return self.Cx, self.Cu




class EndEffConstraintModel():
    def __init__(self, pin_robot, lxmin, lxmax):
        self.lxmin = lxmin
        self.lxmax = lxmax

        self.pin_robot = pin_robot

        self.lumax = np.array([300, 300, 100, 100, 50, 50, 20])
        self.lumin = - self.lumax
        self.frame_id = pin_robot.model.getFrameId("contact")
      
        
        self.Cu = np.eye(self.pin_robot.nq)

        self.ncx = len(lxmin)
        self.ncu = self.pin_robot.nq

    def calc(self, x, u=None): 
        q = x[:self.pin_robot.nq]

        pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)

        p = get_p(q, self.pin_robot, self.frame_id)
        J = pin.computeFrameJacobian(self.pin_robot.model, self.pin_robot.data, q, self.frame_id, pin.LOCAL_WORLD_ALIGNED)[:3]

        c = p 

        return c, u

    def calcDiff(self, x, u=None): 
        q = x[:self.pin_robot.nq]
        pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)

        J = pin.computeFrameJacobian(self.pin_robot.model, self.pin_robot.data, q, self.frame_id, pin.LOCAL_WORLD_ALIGNED)[:3]

        Cx = np.zeros((self.ncx, 2*self.pin_robot.nq))
        Cx[:, :self.pin_robot.nq] = J
        return Cx, self.Cu