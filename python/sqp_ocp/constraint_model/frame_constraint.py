## This enforces box constraints on the give constraint
## Author : Armand Jordana
## Date : 12/04/2023

import pinocchio as pin
import numpy as np
from . abstract_model import ConstraintModelAbstact

class EndEffConstraintModel(ConstraintModelAbstact):
    def __init__(self, pin_robot, lmin, lmax, nc, nx, nu):
        ConstraintModelAbstact.__init__(self, nc, nx, nu)
        assert len(lmin) == 3
        assert len(lmin) == 3

        self.lmin = lmin
        self.lmax = lmax
        self.pin_robot = pin_robot
        self.frame_id = pin_robot.model.getFrameId("contact")
        self.nc = nc
        self.Cu = np.zeros((self.nc, nu))
        self.Cx = np.zeros((self.nc, nx))


    def calc(self, cdata, data, x, u=None): 
        # q = x[:self.pin_robot.nq]
        # pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)
        # cdata.c = get_p(q, self.pin_robot, self.frame_id)

        cdata.c = data.differential.pinocchio.oMf[self.frame_id].translation

    def calcDiff(self, cdata, data, x, u=None): 
        J = pin.getFrameJacobian(self.pin_robot.model, data.differential.pinocchio, self.frame_id, pin.LOCAL_WORLD_ALIGNED)[:3]
        
        # q = x[:self.pin_robot.nq]
        # pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)
        # J = pin.computeFrameJacobian(self.pin_robot.model, self.pin_robot.data, q, self.frame_id, pin.LOCAL_WORLD_ALIGNED)[:3]
        
        self.Cx[:, :self.pin_robot.nq] = J.copy()
        cdata.Cx = self.Cx
        cdata.Cu = self.Cu