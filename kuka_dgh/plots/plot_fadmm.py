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
CONFIG_NAME = 'circle_cssqp'
CONFIG_PATH = 'config/'+CONFIG_NAME+".yml"
config      = path_utils.load_yaml_file(CONFIG_PATH)


# Load data 
SIM = True 


# Create data Plottger
s = SimpleDataPlotter()


if(SIM):
    data_path = '/tmp/'
    data_name = 'circle_cssqp_SIM_2023-10-17T16:39:01.063617_cssqp'
else:
    r = DataReader('/home/skleff/Desktop/circle_PROXQP.mds')

r = DataReader(data_path+data_name+'.mds')
N = r.data['absolute_time'].shape[0]
print("Total number of control cycles = ", N)

fig, ax = plt.subplots(7, 1, sharex='col') 
ax[0].plot(r.data['qp_iters'], label='qp iters')
ax[1].plot(r.data['constraint_norm'], label='constraint_norm')
ax[2].plot(r.data['gap_norm'], label='gap_norm')
ax[3].plot(r.data['KKT'], label='KKT residual')
ax[3].plot(N*[config['solver_termination_tolerance']], label= 'KKT residual tolerance', color = 'r')
ax[4].plot(r.data['ddp_iter'], label='# solver iterations')
ax[5].plot(r.data['t_child']*1000, label='OCP solve time')
ax[5].plot(N*[1000./config['ctrl_freq']], label= 'dt_MPC', color='r')
ax[6].plot((r.data['timing_control'])* 1000, label='Control cycle time')
ax[6].plot(N*[1000./config['ctrl_freq']], label= 'dt_MPC', color='r')
for i in range(len(ax)):
    ax[i].legend()
    

# Limits
xlb = config['stateLowerLimit']
xub = config['stateUpperLimit']

qlb = np.array([xlb[:nq]]*N) ; qub = np.array([xub[:nq]]*N)
vlb = np.array([xlb[nq:]]*N) ; vub = np.array([xub[nq:]]*N)

eps = 0.05

s.plot_joint_pos( [r.data['joint_positions'], qlb, qub], 
                   ['CSSQP', 'lb', 'ub'], 
                   ['b', 'k', 'k'], 
                   linestyle=['solid','solid', 'dotted', 'dotted'],
                   ylims=[model.lowerPositionLimit-eps, model.upperPositionLimit+eps] )
# s.plot_joint_vel( [r.data['joint_velocities'],  vlb, vub], 
#                   ['CSSQP', 'lb', 'ub'], 
#                   ['b', 'k', 'k'], 
#                   linestyle=['solid','solid', 'dotted', 'dotted'],
#                   ylims=[-model.velocityLimit-eps, +model.velocityLimit+eps] )


# # # For SIM robot only
# if(SIM):
#     s.plot_joint_tau( [r.data['tau']], 
#                     #    -config['ctrlLimit'], 
#                     #    config['ctrlLimit']],
#                     #    r.data['tau_riccati'], 
#                     #    r.data['tau_gravity']], 
#                       ['CSSQP', 'limit'], 
#                       ['r', 'g', 'b', [0.2, 0.2, 0.2, 0.5]],
#                       ylims=[-model.effortLimit, +model.effortLimit] )
# # For REAL robot only !! DEFINITIVE FORMULA !!
# else:
#     # Our self.tau was subtracted gravity, so we add it again
#     # joint_torques_measured DOES include the gravity torque from KUKA
#     # There is a sign mismatch in the axis so we use a minus sign
#     s.plot_joint_tau( [-r.data['joint_cmd_torques'], 
#                        r.data['joint_torques_measured'], 
#                        r.data['tau'] + r.data['tau_gravity']], 
#                   ['-cmd (FRI)', 'Measured', 'Desired (+g(q))', 'Measured - EXT'], 
#                   [[0.,0.,0.,0.], 'g', 'b', 'y'],
#                   ylims=[-model.effortLimit, +model.effortLimit] )
#     s.plot_joint_tau([-r.data['joint_cmd_torques'],  
#                       r.data['joint_torques_measured'],  
#                       r.data['tau']], labels=['cmd', 'mea', 'sent'], 
#                       colors=['k', 'g', 'b'])



p_mea = get_p_(r.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))
# p_des = get_p_(r.data['x_des'][:,:nq], pinrobot.model, pinrobot.model.getFrameId('contact'))
target_position = np.zeros((N, 3)) #r.data['target_position'] #
target_position[:,0] = r.data['target_position_x'][:,0]
target_position[:,1] = r.data['target_position_y'][:,0]
target_position[:,2] = r.data['target_position_z'][:,0]
s.plot_ee_pos( [p_mea, 
                target_position],  
               ['CSSQP', 'ref (position cost)'], 
               ['b', 'k', 'g'], 
            #    ylims=[config['stateLowerLimit'][:nq], config['stateUpperLimit'][:nq]],
               linestyle=['solid','solid', 'dotted', 'solid'])

# v_mea = get_v_(r.data['joint_positions'], r.data['joint_velocities'], pinrobot.model, pinrobot.model.getFrameId('contact'))
# s.plot_ee_vel( [v_mea], 
                  
#                ['CSSQP'], 
#                ['r', 'b'], 
#             #    ylims=[config['stateLowerLimit'][nq:], config['stateUpperLimit'][nq:]],
#                linestyle=['solid','solid'])

plt.show()