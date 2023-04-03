import numpy as np



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