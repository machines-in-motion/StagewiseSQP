import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'python'
os.sys.path.insert(1, str(python_path))

import crocoddyl
import numpy as np
import pinocchio as pin
np.set_printoptions(precision=4, linewidth=180)
import example_robot_data as robex
import sobec

np.random.seed(10)
TOL = 1e-3


# Numerical difference function
def numdiff(f,inX,h=1e-6):
    f0 = f(inX).copy()
    x = inX.copy()
    Fx = []
    for ix in range(len(x)):
        x[ix] += h
        Fx.append((f(x)-f0)/h)
        x[ix] = inX[ix]
    return np.array(Fx).T


class Force3DConstraintModel(crocoddyl.ConstraintModelAbstract):
    def __init__(self, state, Fmin, Fmax, nc, nx, nu):
        crocoddyl.ConstraintModelAbstract.__init__(self, state, nc, nu, Fmin, Fmax, 'force')
        self.lmin = Fmin
        self.lmax = Fmax
        
    def calc(self, cdata, data, x, u=None): 
        cdata.c = data.differential.pinocchio.lambda_c[:3]

    def calcDiff(self, cdata, data, x, u=None):
        cdata.Cx = data.differential.df_dx[:3]
        cdata.Cu = data.differential.df_du[:3]



# Cone constraint
class FrictionConeConstraint(crocoddyl.ConstraintModelAbstract):
    def __init__(self, state, mu, nc, nu, frameId, pinRefFrame):
        crocoddyl.ConstraintModelAbstract.__init__(self, state, nc, nu, np.array([0.]), np.array([np.inf]), 'friction')
        self.lmin = np.array([0.])
        self.lmax = np.array([np.inf])
        self.mu = mu
        self.dcone_df = np.zeros((1, 3))
        self.frameId = frameId
        self.pinRefFrame = pinRefFrame
        # self.nc = nc 
        assert nc == 1 

    def calc(self, cdata, data, x, u=None): 
        # constraint residual (expressed in constraint ref frame already)
        F = data.differential.pinocchio.lambda_c[:3]
        cdata.c = np.array([- self.mu * F[2] - np.sqrt(F[0]**2 + F[1]**2)])

    def calcDiff(self, cdata, data, x, u=None):
        F = data.differential.pinocchio.lambda_c[:3]
        self.dcone_df[0, 0] = - F[0] / np.sqrt(F[0]**2 + F[1]**2)
        self.dcone_df[0, 1] = - F[1] / np.sqrt(F[0]**2 + F[1]**2)
        self.dcone_df[0, 2] = - self.mu
        cdata.Cx = self.dcone_df @ data.differential.df_dx[:3] 
        cdata.Cu = self.dcone_df @ data.differential.df_du[:3]

# Load robot : works with talos arm, not with kinova
robot_name = 'talos_arm'# 'kinova'  #'talos_arm'
contactFrameName = 'wrist_left_ft_tool_link'  #'j2s6s200_joint_finger_tip_1' # 'wrist_left_ft_tool_link'
robot = robex.load(robot_name)
robot.data = robot.model.createData()
# Initial conditions
q0 = pin.randomConfiguration(robot.model) 
v0 = np.random.rand(robot.model.nv) 
a0 = np.random.rand(robot.model.nv) 
tau0 = np.random.rand(robot.model.nv) 
x0 = np.concatenate([q0, v0])
# BG gains
gains = np.zeros(2)
nq = robot.model.nq
nv = robot.model.nv
nc = 1
pinRefFrame = pin.LOCAL_WORLD_ALIGNED
contact_frame_id = robot.model.getFrameId(contactFrameName)
pin.forwardKinematics(robot.model, robot.data, q0, v0, a0)
pin.updateFramePlacements(robot.model, robot.data)


state = crocoddyl.StateMultibody(robot.model)
actuation = crocoddyl.ActuationModelFull(state)
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)
contactModel = sobec.ContactModelMultiple(state, actuation.nu)
baumgarte_gains  = np.array([0., 50.])
contactItem = sobec.ContactModel3D(state, contact_frame_id, robot.data.oMf[contact_frame_id].translation, baumgarte_gains, pinRefFrame) 
contactModel.addContact("contact", contactItem, active=True)
uResidual = crocoddyl.ResidualModelContactControlGrav(state)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
xResidual = crocoddyl.ResidualModelState(state, x0 )
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
desired_wrench = np.array([0., 0., -100., 0., 0., 0.])
frameForceResidual = sobec.ResidualModelContactForce(state, contact_frame_id, pin.Force(desired_wrench), 3, actuation.nu)
contactForceCost = crocoddyl.CostModelResidual(state, frameForceResidual)
runningCostModel.addCost("stateReg", xRegCost, 1e-2)
runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
runningCostModel.addCost("force", contactForceCost, 10.)
terminalCostModel.addCost("stateReg", xRegCost, 1e-2)
DAM = sobec.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, runningCostModel, inv_damping=0., enable_force=True) #, contact_frame_id)
# jMf = robot.model.frames[contact_frame_id].placement
IAM = crocoddyl.IntegratedActionModelEuler(DAM, 1e-2)
IAD = IAM.createData()


# TEST FORCE CONSTRAINT IN CPP AND PYTHON
clip_force_min = np.array([-np.inf, -np.inf, 5])
clip_force_max = np.array([np.inf, np.inf, np.inf]) 

# CPP model
forcemodel_cpp = crocoddyl.ContactForceConstraintModel3D(state, actuation.nu, contact_frame_id, clip_force_min, clip_force_max, "forceConstraint", pinRefFrame)
forcemodel_cpp.contact_dynamics_ref = pinRefFrame
CM_CPP = forcemodel_cpp
CD_CPP = CM_CPP.createData()
IAD_CPP = IAM.createData()
    # Action models calc & calcDiff
IAM.differential.calc(IAD_CPP.differential, x0, tau0)
IAM.differential.calcDiff(IAD_CPP.differential, x0, tau0)
IAM.calc(IAD_CPP, x0, tau0)
IAM.calcDiff(IAD_CPP, x0, tau0)
    # Constraint model calc & calcDiff
CM_CPP.calc(CD_CPP, IAD_CPP, x0, tau0)
CM_CPP.calcDiff(CD_CPP, IAD_CPP, x0, tau0)

# Python model
forcemodel_py = Force3DConstraintModel(state, clip_force_min, clip_force_max, 3, state.nx, actuation.nu) 
CM_PY  = forcemodel_py
CD_PY  = CM_PY.createData()
    # Action models calc & calcDiff
IAD_PY = IAM.createData()
IAM.differential.calc(IAD_PY.differential, x0, tau0)
IAM.differential.calcDiff(IAD_PY.differential, x0, tau0)
IAM.calc(IAD_PY, x0, tau0)
IAM.calcDiff(IAD_PY, x0, tau0)
    # Constraint model calc & calcDiff
CM_PY.calc(CD_PY, IAD_PY, x0, tau0)
CM_PY.calcDiff(CD_PY, IAD_PY, x0, tau0)

# # Test that CPP matches Python
assert(np.linalg.norm(CD_CPP.c - CD_PY.c) <= TOL)
assert(np.linalg.norm(CD_CPP.Cx - CD_PY.Cx) <= TOL)
assert(np.linalg.norm(CD_CPP.Cu - CD_PY.Cu) <= TOL)
# Check IAM 
assert(np.linalg.norm(IAD_CPP.xnext - IAD_PY.xnext) <= TOL)
assert(np.linalg.norm(IAD_CPP.Fx - IAD_PY.Fx) <= TOL)
assert(np.linalg.norm(IAD_CPP.Fu - IAD_PY.Fu) <= TOL)

# Test derivatives vs numdiff
def cm_calc(cm, cd, iam, iad, q, v, u):
    iam.calc(iad, np.concatenate([q,v]), u)
    cm.calc(cd, iad, np.concatenate([q,v]), u)
    return cd.c
def iam_calc(iam, iad, q, v, u):
    iam.calc(iad, np.concatenate([q,v]), u)
    return iad.xnext
#  Numdiff IAM CPP
dF_dq_ND_CPP = numdiff(lambda q_:iam_calc(IAM, IAD_CPP, q_, v0, tau0), q0)
dF_dv_ND_CPP = numdiff(lambda v_:iam_calc(IAM, IAD_CPP, q0, v_, tau0), v0)
dF_dtau_ND_CPP = numdiff(lambda tau_:iam_calc(IAM, IAD_CPP, q0, v0, tau_), tau0)
assert(np.linalg.norm(dF_dq_ND_CPP - IAD_CPP.Fx[:,:nq]) <= TOL)
assert(np.linalg.norm(dF_dv_ND_CPP - IAD_CPP.Fx[:,nq:]) <= TOL)
assert(np.linalg.norm(dF_dtau_ND_CPP - IAD_CPP.Fu) <= TOL)
# Numdiff IAM PY
#TODO

# Numdiff constraint PY
dC_dq_ND_PY   = numdiff(lambda q_:cm_calc(CM_PY, CD_PY, IAM, IAD_PY, q_, v0, tau0), q0)
dC_dv_ND_PY   = numdiff(lambda v_:cm_calc(CM_PY, CD_PY, IAM, IAD_PY, q0, v_, tau0), v0)
dC_dtau_ND_PY = numdiff(lambda tau_:cm_calc(CM_PY, CD_PY, IAM, IAD_PY, q0, v0, tau_), tau0)
assert(np.linalg.norm(dC_dq_ND_PY - CD_PY.Cx[:,:nq]) <= TOL)
assert(np.linalg.norm(dC_dv_ND_PY - CD_PY.Cx[:,nq:]) <= TOL)
assert(np.linalg.norm(dC_dtau_ND_PY - CD_PY.Cu) <= TOL)

# Numdiff constraint CPP
dC_dq_ND_CPP   = numdiff(lambda q_:cm_calc(CM_CPP, CD_CPP, IAM, IAD_CPP, q_, v0, tau0), q0)
dC_dv_ND_CPP   = numdiff(lambda v_:cm_calc(CM_CPP, CD_CPP, IAM, IAD_CPP, q0, v_, tau0), v0)
dC_dtau_ND_CPP = numdiff(lambda tau_:cm_calc(CM_CPP, CD_CPP, IAM, IAD_CPP, q0, v0, tau_), tau0)
assert(np.linalg.norm(dC_dq_ND_CPP - CD_CPP.Cx[:,:nq]) <= TOL)
assert(np.linalg.norm(dC_dv_ND_CPP - CD_CPP.Cx[:,nq:]) <= TOL)
assert(np.linalg.norm(dC_dtau_ND_CPP - CD_CPP.Cu) <= TOL)





# TEST FRICTION CONE CONSTRAINT DERIVATIVES
frictionmodel = FrictionConeConstraint(state, 0., 1, actuation.nu, contact_frame_id, pinRefFrame)
def cm_calc(cm, cd, iam, iad, q, v, u):
    iam.calc(iad, np.concatenate([q,v]), u)
    cm.calc(cd, iad, np.concatenate([q,v]), u)
    return cd.c
def iam_calc(iam, iad, q, v, u):
    iam.calc(iad, np.concatenate([q,v]), u)
    return iad.xnext
CM = frictionmodel
CD = CM.createData()
IAM.differential.calc(IAD.differential, x0, tau0)
IAM.differential.calcDiff(IAD.differential, x0, tau0)
IAM.calc(IAD, x0, tau0)
IAM.calcDiff(IAD, x0, tau0)
CM.calc(CD, IAD, x0, tau0)
CM.calcDiff(CD, IAD, x0, tau0)
# Dynamics derivatives
dF_dq = IAD.Fx[:,:nq]
dF_dv = IAD.Fx[:,nq:]
dF_dtau = IAD.Fu
# Constraints derivatives
dC_dq = CD.Cx[:nq]
dC_dv = CD.Cx[nq:]
dC_dtau = CD.Cu
# Numdiff
dF_dq_ND = numdiff(lambda q_:iam_calc(IAM, IAD, q_, v0, tau0), q0)
dF_dv_ND = numdiff(lambda v_:iam_calc(IAM, IAD, q0, v_, tau0), v0)
dF_dtau_ND = numdiff(lambda tau_:iam_calc(IAM, IAD, q0, v0, tau_), tau0)
dC_dq_ND = numdiff(lambda q_:cm_calc(CM, CD, IAM, IAD, q_, v0, tau0), q0)
dC_dv_ND = numdiff(lambda v_:cm_calc(CM, CD, IAM, IAD, q0, v_, tau0), v0)
dC_dtau_ND = numdiff(lambda tau_:cm_calc(CM, CD, IAM, IAD, q0, v0, tau_), tau0)
assert(np.linalg.norm(dF_dq - dF_dq_ND) <= TOL)
assert(np.linalg.norm(dF_dv - dF_dv_ND) <= TOL)
assert(np.linalg.norm(dF_dtau - dF_dtau_ND) <= TOL)
assert(np.linalg.norm(dC_dq - dC_dq_ND) <= TOL)
assert(np.linalg.norm(dC_dv - dC_dv_ND) <= TOL)
assert(np.linalg.norm(dC_dtau - dC_dtau_ND) <= TOL)
