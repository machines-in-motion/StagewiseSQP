from plot_utils import SimpleDataPlotter
from mim_data_utils import DataReader
from core_mpc.pin_utils import *
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt 
from core_mpc import path_utils


from robot_properties_kuka.config import IiwaConfig
pinrobot = IiwaConfig.buildRobotWrapper()
model = pinrobot.model
data = model.createData()
frameId = model.getFrameId('contact')
nq = model.nq ; nv = model.nv ; nc = 3
# Overwrite effort limit as in DGM
model.effortLimit = np.array([100, 100, 50, 50, 20, 10, 10])



# Load config file
CONFIG_NAME = 'kuka_circle_SQP'
CONFIG_PATH = "demos/"+CONFIG_NAME+".yml"
config      = path_utils.load_yaml_file(CONFIG_PATH)


# Load data 
SIM = False 


# Create data Plottger
s = SimpleDataPlotter()


if(SIM):
    # r = DataReader('/tmp/kuka_reach_SQP_sim_SQP.mds')
    # r = DataReader('/tmp/kuka_reach_SQP_sim_FDDP.mds')
    r = DataReader('/home/skleff/ws/workspace/src/SQP/data/kuka_circle_SQP_sim_SQP.mds')
else:
    r2 = DataReader('/home/skleff/ws/workspace/src/SQP/data/circle_SQP.mds')
    r = DataReader('/home/skleff/ws/workspace/src/SQP/data/circle_FDDP.mds')

N = r.data['tau'].shape[0]


# fig, ax = plt.subplots(4, 1, sharex='col') 
# ax[0].plot(r.data['count']-1, label='count')
# ax[1].plot(r.data['t_child'], label='child')
# ax[1].plot(r.data['t_child_1'], label='child_1 (not solve)')
# ax[2].plot(r.data['ddp_iter'], label='iter')
# ax[3].plot(r.data['t_run'], label='t_run')
# ax[1].plot(N*[1./config['plan_freq']], label= 'mpc')
# ax[3].plot(N*[1./config['plan_freq']], label= 'mpc')
# # handles, labels = ax[0].get_legend_handles_labels()
# fig.legend() #handles, labels, loc='upper right', prop={'size': 16})




s.plot_joint_pos( [r.data['joint_positions'], r2.data['joint_positions']], 
                   ['fddp', 'SQP'], 
                   ['r', 'b'], 
                   ylims=[model.lowerPositionLimit, model.upperPositionLimit] )
s.plot_joint_vel( [r.data['joint_velocities'], r2.data['joint_velocities']], 
                  ['fddp', 'SQP'], 
                  ['r', 'b'], 
                  ylims=[-model.velocityLimit, +model.velocityLimit] )
# s.plot_joint_vel( [r.data['joint_accelerations']], ['mea'], ['r'],)

# For SIM robot only
if(SIM):
    s.plot_joint_tau( [r.data['tau'], r.data['tau_ff'], r.data['tau_riccati'], r.data['tau_gravity']], 
                      ['total', 'ff', 'riccati', 'gravity'], 
                      ['r', 'g', 'b', [0.2, 0.2, 0.2, 0.5]],
                      ylims=[-model.effortLimit, +model.effortLimit] )
# For REAL robot only !! DEFINITIVE FORMULA !!
else:
    # Our self.tau was subtracted gravity, so we add it again
    # joint_torques_measured DOES include the gravity torque from KUKA
    # There is a sign mismatch in the axis so we use a minus sign
    s.plot_joint_tau( [-r.data['joint_cmd_torques'], r.data['joint_torques_measured'], r.data['tau'] + r.data['tau_gravity']], 
                  ['-cmd (FRI)', 'Measured', 'Desired (+g(q))', 'Measured - EXT'], 
                  [[0.,0.,0.,0.], 'g', 'b', 'y'],
                  ylims=[-model.effortLimit, +model.effortLimit] )


p_mea = get_p_(r.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))
p_mea2 = get_p_(r2.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))
p_des = get_p_(r.data['x_des'][:,:nq], pinrobot.model, pinrobot.model.getFrameId('contact'))
target_position = np.zeros((N, 3)) #r.data['target_position'] #
target_position[:,0] = r.data['target_position_x'][:,0]
target_position[:,1] = r.data['target_position_y'][:,0]
target_position[:,2] = r.data['target_position_z'][:,0]
s.plot_ee_pos( [p_mea, 
                p_mea2,
                target_position],  
               ['fddp', 'SQP', 'ref (position cost)'], 
               ['r', 'b', 'k', 'g'], 
               linestyle=['solid','solid', 'dotted', 'solid'])

v_mea = get_v_(r.data['joint_positions'], r.data['joint_velocities'], pinrobot.model, pinrobot.model.getFrameId('contact'))
v_mea2 = get_v_(r2.data['joint_positions'], r2.data['x_des'][:,nq:nq+nv], pinrobot.model, pinrobot.model.getFrameId('contact'))
s.plot_ee_vel( [v_mea, 
                v_mea2],  
               ['fddp', 'SQP'], 
               ['r', 'b'], 
               linestyle=['solid','solid'])