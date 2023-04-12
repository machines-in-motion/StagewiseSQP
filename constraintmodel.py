import numpy as np
from pin_utils import get_p
import pinocchio as pin


class ConstraintModelAbstact():
    def __init__(self, nc, nx, nu):
        self.nc = nc
        self.nx = nx
        self.nu = nu

    def createData(self):
        data = ConstraintData(self)
        return data
        

class ConstraintData():
    def __init__(self, cmodel):
        self.c = np.zeros(cmodel.nc)
        self.Cx = np.zeros((cmodel.nc, cmodel.nx))
        self.Cu = np.zeros((cmodel.nc, cmodel.nu))


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


class Force6DConstraintModel(ConstraintModelAbstact):
    def __init__(self, Fmin, Fmax, nc, nx, nu):
        ConstraintModelAbstact.__init__(self, nc, nx, nu)
        self.lmin = Fmin
        self.lmax = Fmax

        self.nc = 6

    def calc(self, cdata, data, x, u=None): 
        cdata.c = data.differential.pinocchio.lambda_c

    def calcDiff(self, cdata, data, x, u=None):
        cdata.Cx = data.differential.df_dx
        cdata.Cu = data.differential.df_du



class LocalCone(ConstraintModelAbstact):
    def __init__(self, mu, nc, nx, nu):
        ConstraintModelAbstact.__init__(self, nc, nx, nu)
        self.lmin = np.array([0.])
        self.lmax = np.array([np.inf])
        self.mu = mu
        self.dcone_df = np.zeros((1, 3))

        self.nc = nc 
        assert nc == 1 

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


class NoConstraint(ConstraintModelAbstact):
    def __init__(self, nx, nu):
        ConstraintModelAbstact.__init__(self, 0, nx, nu)
        self.nc = 0
        self.Cx = np.zeros((self.nc, nx))
        self.Cu = np.zeros((self.nc, nu))
        self.lmin = np.array([])
        self.lmax = np.array([])

    def calc(self, cdata, data, x, u=None): 
        cdata.c = np.array([])

    def calcDiff(self, cdata, data, x, u=None):
        cdata.Cx = self.Cx
        cdata.Cu = self.Cu

class ConstraintModel(ConstraintModelAbstact):
    def __init__(self, constraintmodels, nx, nu):
        self.nc = sum([cmodel.nc for cmodel in constraintmodels])
        ConstraintModelAbstact.__init__(self, self.nc, nx, nu)
        self.lmin = np.concatenate([cmodel.lmin for cmodel in constraintmodels])
        self.lmax = np.concatenate([cmodel.lmax for cmodel in constraintmodels])
    
        self.cdatas = [cmodel.createData() for cmodel in constraintmodels]
        self.cmodels = constraintmodels

    def calc(self, cdata, data, x, u=None): 
        for (ci, di) in zip(self.cmodels, self.cdatas):
            ci.calc(di, data, x, u)
        cdata.c = np.concatenate([cdata.c for cdata in self.cdatas])

    def calcDiff(self, cdata, data, x, u=None):
        for (ci, di) in zip(self.cmodels, self.cdatas):
            ci.calcDiff(di, data, x, u)
        cdata.Cx = np.concatenate([cdata.Cx for cdata in self.cdatas])    
        cdata.Cu = np.concatenate([cdata.Cu for cdata in self.cdatas])    
    
    
    def compute_CxCu(self, dx, du):
        """returns Cx @ dx , Cu @ du"""
        z = np.zeros(self.nc)
        count = 0
        for (ci, di) in zip(self.cmodels, self.cdatas):
            z[count: count + ci.nc] = di.Cx @ dx + di.Cu @ du
            count += ci.nc
        return z

    def compute_transpose(self, vec):
        """returns Cx.T @ vec , Cu.T @ vec"""
        pass

    def compute_ADMM_hessian(self, rho_vec):
        """returns Cx.T @ rho_mat@ Cx, Cx.T @ rho_mat@ Cu, Cu.T @ rho_mat@ Cu ,"""
        pass