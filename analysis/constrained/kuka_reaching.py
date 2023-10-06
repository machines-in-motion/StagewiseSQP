### Kuka reaching example with different constraint implementation

import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'python'
os.sys.path.insert(1, str(python_path))

import crocoddyl
import numpy as np
import pinocchio as pin
np.set_printoptions(precision=4, linewidth=180)
from friction_cone import FrictionConeConstraint


# # # # # # # # # # # # #
### LOAD ROBOT MODEL  ###
# # # # # # # # # # # # #

# Or use robot_properties_kuka 
from robot_properties_kuka.config import IiwaConfig
robot = IiwaConfig.buildRobotWrapper()

model = robot.model
nq = model.nq; nv = model.nv; nu = nq; nx = nq+nv
q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.])
v0 = np.zeros(nv)
x0 = np.concatenate([q0, v0]).copy()
robot.framesForwardKinematics(q0)
robot.computeJointJacobians(q0)
fid = model.getFrameId("contact")

# # # # # # # # # # # # # # #
###  SETUP CROCODDYL OCP  ###
# # # # # # # # # # # # # # #

# State and actuation model
state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFull(state)

# Running and terminal cost models
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)


# Create cost terms 
  # Control regularization cost
uResidual = crocoddyl.ResidualModelContactControlGrav(state)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
  # State regularization cost
xResidual = crocoddyl.ResidualModelState(state, x0)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
  # endeff frame translation cost
endeff_frame_id = model.getFrameId("contact")
# endeff_translation = robot.data.oMf[endeff_frame_id].translation.copy()
endeff_translation = np.array([0.7, 0, 1.1]) # move endeff +30 cm along x in WORLD frame
frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, endeff_frame_id, endeff_translation)
frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)


# Add costs
runningCostModel.addCost("stateReg", xRegCost, 1e-1)
runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
runningCostModel.addCost("translation", frameTranslationCost, 1)
terminalCostModel.addCost("stateReg", xRegCost, 1e-1)
terminalCostModel.addCost("translation", frameTranslationCost, 1)

# Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
running_DAM = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, crocoddyl.ContactModelMultiple(state, actuation.nu), runningCostModel)
terminal_DAM = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, crocoddyl.ContactModelMultiple(state, actuation.nu), terminalCostModel)

# Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
dt = 1e-3
runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)

# Optionally add armature to take into account actuator's inertia
# runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
# terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])

# Create the shooting problem
T = 100
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)



# choose scenario: 0 or 1 or 2 or 3
option = 0

# State (joint all vel box) and control constraints (torque axis 2)
if option == 0:    
  clip_state_max = np.array([np.inf]*14)
  clip_state_min = -np.array([np.inf]*7 + [np.inf]*7)
  clip_ctrl = np.array([np.inf, np.inf , np.inf, np.inf, np.inf, np.inf , np.inf] )
  statemodel = crocoddyl.StateConstraintModel(state, 7, clip_state_min, clip_state_max, 'stateConstraint')
  controlmodel = crocoddyl.ControlConstraintModel(state, 7,  -clip_ctrl, clip_ctrl, 'ctrlConstraint')
  nc = statemodel.nc + controlmodel.nc
  ConstraintModel = crocoddyl.ConstraintStack([statemodel, controlmodel], state, nc, 7, 'runningConstraint')
  clip_state_end = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf , np.inf] + [0.0]*7)
  terminal = crocoddyl.StateConstraintModel(state, 7, -clip_state_end, clip_state_end, 'stateConstraint')
  TerminalConstraintModel = crocoddyl.ConstraintStack([terminal], state, 14, 7, 'terminalConstraint')
  constraintModels =[controlmodel] * (T) + [terminal]

  # # Plot results 
  # p_mea1 = get_p_(r1.data['joint_positions'][N_start:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
  # p_mea2 = get_p_(r2.data['joint_positions'][N_start:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
  # fig_circle, ax_circle = plot_endeff_yz(p_mea2, target_position, "Constrained") 
  # ax_circle.plot(p_mea1[:,1], p_mea1[:,2], color='g', linewidth=4, label='Unconstrained', alpha=0.5) 
  # ax_circle.set_xlim(-0.33, +0.33)
  # ax_circle.set_ylim(0.15, 0.8)
  # ax_circle.plot(p_mea2[0,1], p_mea2[0,2], 'ro', markersize=16)
  # ax_circle.text(0., 0.1, '$x_0$', fontdict={'size':26})
  # handles, labels = ax_circle.get_legend_handles_labels()
  # fig_circle.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.12, 0.885), prop={'size': 26}) 
  # fig_circle.savefig('/home/skleff/data_paper_CSSQP/jointpos_circle_plot.pdf', bbox_inches="tight")
  # # Joint pos

# Only state constraints
elif option == 1:    
  clip_state_max = np.array([np.inf]*14)
  clip_state_min = -np.array([np.inf]*7 + [0.5]*7)
  statemodel = crocoddyl.StateConstraintModel(state, 7, clip_state_min, clip_state_max, 'stateConstraint')
  clip_state_end = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf , np.inf] + [0.00]*7)
  TerminalConstraintModel = crocoddyl.StateConstraintModel(state, 7, -clip_state_end, clip_state_end, 'stateConstraint')
  constraintModels =  [crocoddyl.NoConstraintModelModel(state, 7)] + [statemodel] * (T-1) + [TerminalConstraintModel]

# End-effector constraints
elif option == 2:
  endeff_translation = np.array([0.7, 0, 1.1]) # move endeff +30 cm along x in WORLD frame
  lmin = np.array([-np.inf, endeff_translation[1], endeff_translation[2]])
  lmax =  np.array([np.inf, endeff_translation[1], endeff_translation[2]])
  constraintModels = [crocoddyl.NoConstraintModelModel(state, 7)] + [crocoddyl.FrameTranslationConstraintModel(state, 7, fid, lmin, lmax)] * T

# # Force constraint
# elif option == 3:
#   clip_force_min = np.array([np.inf]*3)
#   clip_force_max = -np.array([np.inf, np.inf, 30.])
#   clip_ctrl = np.array([np.inf, 40 , np.inf, np.inf, np.inf, np.inf , np.inf] )
#   clip_state_max = np.array([np.inf]*14)
#   clip_state_min = -np.array([np.inf]*14)
#   problem.runningModels[0].differential.
#   statemodel = crocoddyl.StateConstraintModel(state, 7, clip_state_min, clip_state_max, 'stateConstraint')
#   controlmodel = crocoddyl.ControlConstraintModel(state, 7,  -clip_ctrl, clip_ctrl, 'ctrlConstraint')
#   forcemodel = crocoddyl.ContactForceConstraintModel3D(state, actuation.nu, fid, clip_force_min, clip_force_max, "forceConstraint", pin.LOCAL_WORLD_ALIGNED)
#   constraintModel = crocoddyl.ConstraintStack([forcemodel, statemodel], state, statemodel.nc + forcemodel.nc, 7, 'runningConstraint')
#   constraintModels =  [constraintModel]*T + [constraintModel] #+ [forcemodel] * (T-1) + [statemodel]

# No constraints
elif option == 4:
  constraintModels = [crocoddyl.NoConstraintModelModel(state, 7)] * (T+1)


xs = [x0] * (T+1)
us = [np.zeros(nu)] * T 



ddp = crocoddyl.SolverFADMM(problem, constraintModels)
# ddp = crocoddyl.SolverPROXQP(problem, constraintModels)
# ddp = crocoddyl.SolverSQP(problem)
# ddp = CSSQP(problem, constraintModels, "ProxQP")
# ddp = CSSQP(problem, constraintModels, "OSQP")
qp_iters = 10000
sqp_ites = 500
ddp.with_callbacks = True
ddp.use_filter_ls = True
ddp.filter_size = 100
ddp.termination_tol = 1e-6
ddp.warm_start = True
ddp.max_qp_iters = qp_iters
# ddp.calc()
import time
# t1 = time.time()
ddp.solve(xs, us, sqp_ites)
# t2 = time.time()
# print(t2 - t1)
# print("after warm start")
# ddp.solve(ddp.xs, ddp.us, sqp_ites)



# Generate plot of number of iterations for each problem
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
 

# Plot end-effector trajectory (y,z) plane
# 
def plot_joint_velocity(ddpSolver, fig=None, ax=None, label=None):
  x = np.array(ddpSolver.xs)
  q = x[:,nq:] 
  # Plots
  tspan = np.linspace(0, T*dt, T+1)
  if(ax is None or fig is None):
      fig, ax = plt.subplots(nq, 1, sharex='col') 
  if(label is None):
      label='Position'
  for i in range(nq):
      # Plot positions
      ax[i].plot(tspan, q[:,i], color='b', linewidth=4, label=None, alpha=0.5)
      ax[i].set_ylabel('$v_%s$'%i, fontsize=20)
      # ax[i].set_xlim(, +0.33)
      ax[i].set_ylim(-model.velocityLimit[i],+model.velocityLimit[i])
      # ax[i].yaxis.set_major_locator(plt.MaxNLocator(2))
      # ax[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
      ax[i].grid(True)
      ax[i].tick_params(axis = 'y', labelsize=18)
      ax[i].tick_params(axis = 'x', labelsize=18)
  # Common x-labels + align
  ax[-1].set_xlabel('Time (s)', fontsize=16)

  ax[-1].grid(True) 
  # y axis labels
  # fig.text(0.05, 0.5, 'Joint position (rad)', va='center', rotation='vertical', fontsize=16)
  # fig.text(0.49, 0.5, 'Joint velocity (rad/s)', va='center', rotation='vertical', fontsize=16)
  # fig.text(0.05, 0.5, 'Joint velocity (rad/s)', va='center', rotation='vertical', fontsize=16)
  fig.subplots_adjust(wspace=0.27)
  fig.align_ylabels()

  handles, labels = ax[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.12, 0.885), prop={'size': 26}) 
  fig.savefig('/home/skleff/data_paper_CSSQP/ocp_case1.pdf', bbox_inches="tight")
  return fig, ax 

plot_joint_velocity(ddp)
# def plot_state(jmea, label):
    # fig0, ax0 = plt.subplots(1, 1, figsize=(19.2,10.8))
    # # Measured
    # ax0.plot(xdata, jmea, color='b', linewidth=4, label=label, alpha=0.5) 
    # # Axis label & ticks
    # ax0.set_ylabel('Joint position $q_1$ (rad)', fontsize=26)
    # ax0.set_xlabel('Time (s)', fontsize=26)
    # ax0.tick_params(axis = 'y', labelsize=22)
    # ax0.tick_params(axis = 'x', labelsize=22)
    # ax0.grid(True) 
    # return fig0, ax0
plt.show()
plt.close('all')