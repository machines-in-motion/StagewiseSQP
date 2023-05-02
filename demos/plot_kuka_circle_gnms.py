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
CONFIG_NAME = 'kuka_circle_fadmm'
CONFIG_PATH = "demos/"+CONFIG_NAME+".yml"
config      = path_utils.load_yaml_file(CONFIG_PATH)


# Load data 
SIM = True 


# Create data Plottger
s = SimpleDataPlotter()


if(SIM):
    # r = DataReader('/tmp/kuka_reach_gnms_sim_GNMS.mds')
    # r = DataReader('/tmp/kuka_reach_gnms_sim_FDDP.mds')
    r = DataReader('/tmp/kuka_circle_sim_FADMM.mds')
else:
    r2 = DataReader('/home/skleff/ws/workspace/src/gnms/data/circle_GNMS.mds')
    r = DataReader('/home/skleff/ws/workspace/src/gnms/data/circle_FDDP.mds')

N = r.data['tau'].shape[0]


fig, ax = plt.subplots(4, 1, sharex='col') 
ax[0].plot(r.data['count']-1, label='count')
ax[1].plot(r.data['t_child'], label='child')
ax[1].plot(r.data['t_child_1'], label='child_1 (not solve)')
ax[2].plot(r.data['ddp_iter'], label='iter')
ax[3].plot(r.data['t_run'], label='t_run')
ax[1].plot(N*[1./config['plan_freq']], label= 'mpc')
ax[3].plot(N*[1./config['plan_freq']], label= 'mpc')
# handles, labels = ax[0].get_legend_handles_labels()
fig.legend() #handles, labels, loc='upper right', prop={'size': 16})




s.plot_joint_pos( [r.data['joint_positions']], #, r2.data['joint_positions']], 
                   ['fddp'], #, 'gnms'], 
                   ['r'], #, 'b'], 
                   ylims=[model.lowerPositionLimit, model.upperPositionLimit] )
s.plot_joint_vel( [r.data['joint_velocities']], #, r2.data['joint_velocities']], 
                  ['fddp'], #, 'gnms'], 
                  ['r'], #, 'b'], 
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
               ['fddp', 'gnms', 'ref (position cost)'], 
               ['r', 'b', 'k', 'g'], 
               linestyle=['solid','solid', 'dotted', 'solid'])

v_mea = get_v_(r.data['joint_positions'], r.data['joint_velocities'], pinrobot.model, pinrobot.model.getFrameId('contact'))
v_mea2 = get_v_(r2.data['joint_positions'], r2.data['x_des'][:,nq:nq+nv], pinrobot.model, pinrobot.model.getFrameId('contact'))
s.plot_ee_vel( [v_mea, 
                v_mea2],  
               ['fddp', 'gnms'], 
               ['r', 'b'], 
               linestyle=['solid','solid'])
# target_force_3d = np.zeros((N, 3))
# target_force_3d[:,0] = -r.data['target_force'][:,0]*0
# target_force_3d[:,1] = r.data['target_force'][:,0]*0
# target_force_3d[:,2] = r.data['target_force'][:,0]
# s.plot_soft_contact_force([r.data['contact_force_3d_measured'], #f_mea_new, #
#                            target_force_3d,
#                            r.data['fpred']],
#                           ['Measured', 'Reference', 'Predicted'], 
#                           ['r', 'k', 'b', 'g'],
#                           linestyle=['solid', 'dotted', 'solid'])




# v_mea = get_v_(r.data['joint_velocities'], r.data['x_des'][:,nq:nq+nv], pinrobot.model, pinrobot.model.getFrameId('contact'))
# v_des = get_v_(r.data['joint_velocities'], r.data['x_des'][:,nq:nq+nv], pinrobot.model, pinrobot.model.getFrameId('contact'))
# s.plot_ee_vel( [v_mea, v_des, np.zeros(v_des.shape)],  
#                ['mea', 'pred', 'ref'], 
#                ['r', 'b', 'k'])


# rpy_mea = get_rpy_(r.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))
# rpy_des = get_rpy_(r.data['x_des'][:,:nq], pinrobot.model, pinrobot.model.getFrameId('contact'))
# s.plot_ee_rpy( [p_mea, p_des, r.data['target_rpy']],  
#                ['mea', 'pred', 'ref'], 
#                ['r', 'b', 'k'])


# w_mea = get_w_(r.data['joint_velocities'], r.data['x_des'][:,nq:nq+nv], pinrobot.model, pinrobot.model.getFrameId('contact'))
# w_des = get_w_(r.data['joint_velocities'], r.data['x_des'][:,nq:nq+nv], pinrobot.model, pinrobot.model.getFrameId('contact'))
# s.plot_ee_w( [w_mea, w_des, np.zeros(w_des.shape)],  
#                ['mea', 'pred', 'ref'], 
#                ['r', 'b', 'k'])

# # Finite diff for acc
# q = r.data['joint_positions']
# v = r.data['joint_velocities']
# a = np.zeros(v.shape)
# for i in range(v.shape[0]):
#     if i>0:
#         a[i,:] = (v[i,:] - v[i-1,:])/s.dt
# f = []
# for i in range(v.shape[0]):
#     pin.framesForwardKinematics(model, data, q[i])
#     f.append(get_external_joint_torques(data.oMf[frameId], r.data['ft_sensor_wrench'][i], pinrobot))
# tau_ft = get_tau_(q, v, a, f, model)

# # Plot external torques sensed by robot vs external torques due to contact force
# s.plot_joint_tau([r.data['joint_ext_torques'], tau_ft],
#                  ['Measured', 'Estimated'],
#                  ['r', 'b'])


plt.show()






