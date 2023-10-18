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
CONFIG_NAME = 'circle_ssqp'
CONFIG_PATH = "config/"+CONFIG_NAME+".yml"
config      = path_utils.load_yaml_file(CONFIG_PATH)


# Load data 
SIM = True 


# Create data Plottger
s = SimpleDataPlotter()

if(SIM):
    data_path = '/tmp/'
    data_name = 'circle_ssqp_SIM_2023-10-17T15:06:17.954978_sqp' 
    
else:
    data_path = '/tmp/'
    data_name = 'reduced_soft_mpc_contact1d_REAL_2023-10-05T17:24:49.414383_soft_rt'
    
r = DataReader(data_path+data_name+'.mds')
N = r.data['absolute_time'].shape[0]
print("Total number of control cycles = ", N)

fig, ax = plt.subplots(4, 1, sharex='col') 
ax[0].plot(r.data['KKT'], label='KKT residual')
ax[0].plot(N*[config['solver_termination_tolerance']], label= 'KKT residual tolerance', color = 'r')
ax[1].plot(r.data['ddp_iter'], label='# solver iterations')
ax[2].plot(r.data['t_child']*1000, label='OCP solve time')
ax[2].plot(N*[1000./config['ctrl_freq']], label= 'dt_MPC', color='r')
ax[3].plot((r.data['timing_control'])* 1000, label='Control cycle time')
ax[3].plot(N*[1000./config['ctrl_freq']], label= 'dt_MPC', color='r')
for i in range(4):
    ax[i].legend()



# s.plot_joint_pos( [r.data['joint_positions'], r.data['x_des'][:,:nq]], # r.data['x'][:,:nq], r.data['x1'][:,:nq]], 
#                    ['mea', 'pred'], #, 'pred0', 'pred1'], 
#                    ['r', 'b'], #[0.2, 0.2, 0.2, 0.5], 'b', 'g'])
#                 #    markers=[None, None, '.', '.']) 
#                    ylims=[model.lowerPositionLimit, model.upperPositionLimit] )
# s.plot_joint_vel( [r.data['joint_velocities'], r.data['x_des'][:,nq:nq+nv]], # r.data['x'][:,nq:nq+nv], r.data['x1'][:,nq:nq+nv]],
#                   ['mea', 'pred'], # 'pred0', 'pred1'], 
#                   ['r', 'b'], #[0.2, 0.2, 0.2, 0.5], 'b', 'g']) 
#                   ylims=[-model.velocityLimit, +model.velocityLimit] )
# # s.plot_joint_vel( [r.data['joint_accelerations']], ['mea'], ['r'],)

# # For SIM robot only
# if(SIM):
#     s.plot_joint_tau( [r.data['tau'], r.data['tau_ff'], r.data['tau_riccati'], r.data['tau_gravity']], 
#                       ['total', 'ff', 'riccati', 'gravity'], 
#                       ['r', 'g', 'b', [0.2, 0.2, 0.2, 0.5]],
#                       ylims=[-model.effortLimit, +model.effortLimit] )
# # For REAL robot only !! DEFINITIVE FORMULA !!
# else:
#     # Our self.tau was subtracted gravity, so we add it again
#     # joint_torques_measured DOES include the gravity torque from KUKA
#     # There is a sign mismatch in the axis so we use a minus sign
#     s.plot_joint_tau( [-r.data['joint_cmd_torques'], r.data['joint_torques_measured'], r.data['tau'] + r.data['tau_gravity']], 
#                   ['-cmd (FRI)', 'Measured', 'Desired (+g(q))', 'Measured - EXT'], 
#                   [[0.,0.,0.,0.], 'g', 'b', 'y'],
#                   ylims=[-model.effortLimit, +model.effortLimit] )


# p_mea = get_p_(r.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))
# p_des = get_p_(r.data['x_des'][:,:nq], pinrobot.model, pinrobot.model.getFrameId('contact'))
# target_position = r.data['target_position'] #np.zeros((N, 3))
# # target_position[:,0] = r.data['target_position_x'][:,0]
# # target_position[:,1] = r.data['target_position_y'][:,0]
# # target_position[:,2] = r.data['target_position_z'][:,0]
# s.plot_ee_pos( [p_mea, 
#                 target_position],  
#                ['mea', 'ref (position cost)'], 
#                ['r',  'k', 'g'], 
#                linestyle=['solid', 'dotted', 'solid'])

# v_mea = get_v_(r.data['joint_velocities'], r.data['x_des'][:,nq:nq+nv], pinrobot.model, pinrobot.model.getFrameId('contact'))
# v_des = get_v_(r.data['joint_velocities'], r.data['x_des'][:,nq:nq+nv], pinrobot.model, pinrobot.model.getFrameId('contact'))
# target_velocity = np.zeros((N, 3))
# target_velocity[:,0] = r.data['target_velocity_x'][:,0]
# target_velocity[:,1] = r.data['target_velocity_y'][:,0]
# target_velocity[:,2] = r.data['target_velocity_z'][:,0]

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






