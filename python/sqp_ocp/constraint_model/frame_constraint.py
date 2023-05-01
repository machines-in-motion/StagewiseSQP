## This enforces box constraints on the give constraint
## Author : Armand Jordana
## Date : 12/04/2023


try:
    from crocoddyl import FrameTranslationConstraintModel as EndEffConstraintModel
except:

    import pinocchio as pin
    import numpy as np
    from . abstract_model import ConstraintModelAbstract

    class EndEffConstraintModel(ConstraintModelAbstract):
        def __init__(self, state, nu, fid, lmin, lmax):
            ConstraintModelAbstract.__init__(self, state, 3, nu, lmin, lmax)
            self.pin_robot = state.pinocchio
            self.frame_id = fid
            self.Cu = np.zeros((self.nc, nu))
            self.Cx = np.zeros((self.nc, state.nx))


        def calc(self, cdata, data, x, u=None): 

            cdata.c = data.differential.pinocchio.oMf[self.frame_id].translation

        def calcDiff(self, cdata, data, x, u=None): 

            J = pin.getFrameJacobian(self.pin_robot, data.differential.pinocchio, self.frame_id, pin.LOCAL)[:3]        
            self.Cx[:, :self.pin_robot.nv] = data.differential.pinocchio.oMf[self.frame_id].rotation @ J.copy()
            cdata.Cx = self.Cx
            cdata.Cu = self.Cu