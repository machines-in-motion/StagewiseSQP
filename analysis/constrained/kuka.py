"""__init__

License: BSD 3-Clause License
Copyright (C) 2023, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""

import crocoddyl
import numpy as np
import mim_solvers
from mim_robots.robot_loader import load_pinocchio_wrapper
import pinocchio as pin

# # # # # # # # # # # # # # #
###       LOAD ROBOT      ###
# # # # # # # # # # # # # # #

import qp_benchmark

robot = load_pinocchio_wrapper("iiwa", locked_joints = ["A7"])
model = robot.model
nq = model.nq
nv = model.nv
nu = nv
# q0 = np.array([0.1, 0.2, 0., 0., -0.2, 0.2]) # initial p_ee = np.array([0.21, 0.0161691, 1.2752]) 
q0 = np.array([-0.8 ,  0.06 , 2.68 , 1.74 , 0.81,  0.91])
v0 = np.zeros(nv)
x0 = np.concatenate([q0, v0]).copy()

problem = qp_benchmark.create_kuka_problem(x0)
# # # # # # # # # # # # # # # #
# ###  SETUP CROCODDYL OCP  ###
# # # # # # # # # # # # # # # #

# # State and actuation model
# state = crocoddyl.StateMultibody(model)
# actuation = crocoddyl.ActuationModelFull(state)

# # Create cost terms
# # Control regularization cost
# uResidual = crocoddyl.ResidualModelControlGrav(state)
# uRegCost = crocoddyl.CostModelResidual(state, uResidual)
# # State regularization cost
# xResidual = crocoddyl.ResidualModelState(state, x0)
# xRegCost = crocoddyl.CostModelResidual(state, xResidual)
# # endeff frame translation cost
# endeff_frame_id = model.getFrameId("contact")
# endeff_translation = np.array([0.5, 0.1, 0.2]) 
# frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
#     state, endeff_frame_id, endeff_translation
# )
# frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)

# # Create contraint on end-effector (small box around initial EE position)
# frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
#     state, endeff_frame_id, np.zeros(3)
# )
# data = model.createData()
# pin.framesForwardKinematics(model, data, x0[:model.nq])
# p0 = data.oMf[endeff_frame_id].translation
# ee_contraint = crocoddyl.ConstraintModelResidual(
#     state,
#     frameTranslationResidual,
#     p0 - np.array([10.5, 0.5, 10.5]),
#     p0 + np.array([0.5, 10.5, 10.5]),
# )
# # Create the running models
# runningModels = []
# dt = 1e-2
# T = 50
# for t in range(T + 1):
#     runningCostModel = crocoddyl.CostModelSum(state)
#     # Add costs
#     runningCostModel.addCost("stateReg", xRegCost, 1e-1)
#     runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
#     if t != T:
#         runningCostModel.addCost("translation", frameTranslationCost, 4)
#     else:
#         runningCostModel.addCost("translation", frameTranslationCost, 40)
#     # Define contraints
#     constraints = crocoddyl.ConstraintModelManager(state, actuation.nu)
#     if t != 0:
#         constraints.addConstraint("ee_bound", ee_contraint)
#     # Create Differential action model
#     running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
#         state, actuation, runningCostModel, constraints
#     )
#     # Apply Euler integration
#     running_model = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
#     runningModels.append(running_model)        
# problem = crocoddyl.ShootingProblem(x0, runningModels[:-1], runningModels[-1])


# # # # # # # # # # # # #
###     SOLVE OCP     ###
# # # # # # # # # # # # #

# # Define warm start
# xs = [x0] * (T + 1)
# us = [np.zeros(nu)] * T

# Define solver
import pathlib
import os
python_path = pathlib.Path('/home/skleff/libs/mim_solvers/python/').absolute()
os.sys.path.insert(1, str(python_path))
from csqp import CSQP
solver = CSQP(problem, "OSQP") #mim_solvers.SolverCSQP(problem)

MAXITER     = 1     
TOL         = 1e-4 
CALLBACKS   = False
MAX_QP_ITER = 10000
MAX_QP_TIME = 10000
EPS_ABS     = 1e-8
EPS_REL     = 0.
SAVE        = False # Save figure 

solver = CSQP(problem, "OSQP")
solver.xs = [solver.problem.x0] * (solver.problem.T + 1)  
solver.us = solver.problem.quasiStatic([solver.problem.x0] * solver.problem.T)
solver.termination_tolerance = TOL
solver.max_qp_iters = MAX_QP_ITER
solver.eps_abs = EPS_ABS
solver.eps_rel = EPS_REL
solver.equality_qp_initial_guess = False
solver.with_callbacks = CALLBACKS

solver.problem.x0 = x0
solver.xs = [x0] * (solver.problem.T + 1) 
solver.us = solver.problem.quasiStatic([x0] * solver.problem.T)
solver.solve(solver.xs, solver.us, MAXITER, False)
solved = solver.is_optimal

# solver.max_qp_iters = 10000
# solver.eps_abs = 1e-8
# solver.eps_rel = 0.
# solver.equality_qp_initial_guess = False
# solver.termination_tolerance = 1e-4
# solver.with_callbacks = True 
# solver.xs = [solver.problem.x0] * (solver.problem.T + 1)  
# solver.us = solver.problem.quasiStatic([solver.problem.x0] * solver.problem.T)
# Solve
# max_iter = 1 #00
# solver.solve(solver.xs, solver.us, max_iter, False)


# x_traj = np.array(solver.xs)
# u_traj = np.array(solver.us)
# p_traj = np.zeros((len(solver.xs), 3))

# for i in range(T + 1):
#     robot.framesForwardKinematics(x_traj[i, :nq])
#     p_traj[i] = robot.data.oMf[endeff_frame_id].translation

# import matplotlib.pyplot as plt 

# time_lin = np.linspace(0, dt * (T + 1), T+1)

# fig, axs = plt.subplots(nq)
# for i in range(nq):
#     axs[i].plot(time_lin, x_traj[:, i])
# fig.suptitle("State trajectory")


# fig, axs = plt.subplots(nq)
# for i in range(nq):
#     axs[i].plot(time_lin[:-1], u_traj[:, i])
# fig.suptitle("Control trajectory")


# fig, axs = plt.subplots(3)
# for i in range(3):
#     axs[i].plot(time_lin, p_traj[:, i])
#     axs[i].plot(time_lin[-1], endeff_translation[i], "o")
# fig.suptitle("End effector trajectory")
# plt.show()

# # viewer
# WITHDISPLAY = True
# if WITHDISPLAY:
#     import time
#     display = crocoddyl.MeshcatDisplay(robot)
#     display.rate = -1
#     display.freq = 1
#     while True:
#         display.displayFromSolver(solver)
#         time.sleep(1.0)