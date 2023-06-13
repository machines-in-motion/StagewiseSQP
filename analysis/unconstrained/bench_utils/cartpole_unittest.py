import numpy as np
np.set_printoptions(precision=4, linewidth=180)
import crocoddyl
from cartpole_swingup import DifferentialActionModelCartpoleNODIFF, DifferentialActionModelCartpole



TOL = 1e-3

np.random.seed(10)

# Creating the DAM for the cartpole

cartpoleDAM_NODIFF = DifferentialActionModelCartpoleNODIFF()

# Using NumDiff for computing the derivatives. We specify the
# withGaussApprox=True to have approximation of the Hessian based on the
# Jacobian of the cost residuals.
cartpole_NumDiff = crocoddyl.DifferentialActionModelNumDiff(cartpoleDAM_NODIFF, True)
cartpoleData_NumDiff = cartpole_NumDiff.createData()



cartpoleDAM = DifferentialActionModelCartpole()

cartpoleData = cartpoleDAM.createData()



x0 = np.array([0., -3.14/2, 0., 0.]) + 10*np.random.random(4)
tau0 = np.random.random(1)




cartpoleDAM.calc(cartpoleData, x0, tau0)
cartpoleDAM.calcDiff(cartpoleData, x0, tau0)


cartpole_NumDiff.calc(cartpoleData_NumDiff, x0, tau0)
cartpole_NumDiff.calcDiff(cartpoleData_NumDiff, x0, tau0)



assert(np.linalg.norm(cartpoleData_NumDiff.Fx - cartpoleData.Fx)<=TOL)
assert(np.linalg.norm(cartpoleData_NumDiff.Fu - cartpoleData.Fu)<=TOL)
assert(np.linalg.norm(cartpoleData_NumDiff.Lu - cartpoleData.Lu)<=TOL)
assert(np.linalg.norm(cartpoleData_NumDiff.Lx - cartpoleData.Lx)<=TOL)
assert(np.linalg.norm(cartpoleData_NumDiff.Lxx - cartpoleData.Lxx)<=TOL)
assert(np.linalg.norm(cartpoleData_NumDiff.Lxu - cartpoleData.Lxu)<=TOL)
assert(np.linalg.norm(cartpoleData_NumDiff.Luu - cartpoleData.Luu)<=TOL)