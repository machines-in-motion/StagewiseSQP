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
CONFIG_NAME = 'kuka_circle_CSSQP'
CONFIG_PATH = CONFIG_NAME+".yml"
config      = path_utils.load_yaml_file(CONFIG_PATH)


# Load data 
SIM = True 


# Create data Plottger
s = SimpleDataPlotter()


if(SIM):
    # r = DataReader('/home/skleff/Desktop/circle_PROXQP.mds')
    # r = DataReader('/home/skleff/Desktop/circle_CSSQP.mds')
    r = DataReader('/tmp/kuka_circle_sim_CSSQP.mds')
else:
    r = DataReader('/home/skleff/Desktop/circle_PROXQP.mds')
N = r.data['tau'].shape[0]

fig, ax = plt.subplots(5, 1, sharex='col') 
# ax[0].plot(r.data['count']-1, label='count')
ax[0].plot(r.data['qp_iters'], label='qp iters')
# ax[1].plot(r.data['t_child_1'], label='child_1 (not solve)')
ax[2].plot(r.data['constraint_norm'], label='constraint_norm')
ax[3].plot(r.data['gap_norm'], label='gap_norm')
ax[4].plot(r.data['t_run'], label='PROXQP cycle')
ax[1].plot(N*[1./config['plan_freq']], label= 'mpc')
# handles, labels = ax[0].get_legend_handles_labels()
fig.legend() #handles, labels, loc='upper right', prop={'size': 16})


# Limits
xlb = config['stateLowerLimit']
xub = config['stateUpperLimit']

qlb = np.array([xlb[:nq]]*N) ; qub = np.array([xub[:nq]]*N)
vlb = np.array([xlb[nq:]]*N) ; vub = np.array([xub[nq:]]*N)

eps = 0.05

# s.plot_joint_pos( [r.data['joint_positions'], qlb, qub], 
#                    ['CSSQP', 'lb', 'ub'], 
#                    ['b', 'k', 'k'], 
#                    linestyle=['solid','solid', 'dotted', 'dotted'],
#                    ylims=[model.lowerPositionLimit-eps, model.upperPositionLimit+eps] )
s.plot_joint_vel( [r.data['joint_velocities'],  vlb, vub], 
                  ['CSSQP', 'lb', 'ub'], 
                  ['b', 'k', 'k'], 
                  linestyle=['solid','solid', 'dotted', 'dotted'],
                  ylims=[-model.velocityLimit-eps, +model.velocityLimit+eps] )


# # For SIM robot only
if(SIM):
    s.plot_joint_tau( [r.data['tau']], 
                    #    -config['ctrlLimit'], 
                    #    config['ctrlLimit']],
                    #    r.data['tau_riccati'], 
                    #    r.data['tau_gravity']], 
                      ['CSSQP', 'limit'], 
                      ['r', 'g', 'b', [0.2, 0.2, 0.2, 0.5]],
                      ylims=[-model.effortLimit, +model.effortLimit] )
# For REAL robot only !! DEFINITIVE FORMULA !!
else:
    # Our self.tau was subtracted gravity, so we add it again
    # joint_torques_measured DOES include the gravity torque from KUKA
    # There is a sign mismatch in the axis so we use a minus sign
    s.plot_joint_tau( [-r.data['joint_cmd_torques'], 
                       r.data['joint_torques_measured'], 
                       r.data['tau'] + r.data['tau_gravity']], 
                  ['-cmd (FRI)', 'Measured', 'Desired (+g(q))', 'Measured - EXT'], 
                  [[0.,0.,0.,0.], 'g', 'b', 'y'],
                  ylims=[-model.effortLimit, +model.effortLimit] )
    s.plot_joint_tau([-r.data['joint_cmd_torques'],  
                      r.data['joint_torques_measured'],  
                      r.data['tau']], labels=['cmd', 'mea', 'sent'], 
                      colors=['k', 'g', 'b'])



p_mea = get_p_(r.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))
p_des = get_p_(r.data['x_des'][:,:nq], pinrobot.model, pinrobot.model.getFrameId('contact'))
target_position = np.zeros((N, 3)) #r.data['target_position'] #
target_position[:,0] = r.data['target_position_x'][:,0]
target_position[:,1] = r.data['target_position_y'][:,0]
target_position[:,2] = r.data['target_position_z'][:,0]
# s.plot_ee_pos( [p_mea, 
#                 target_position],  
#                ['CSSQP', 'ref (position cost)'], 
#                ['b', 'k', 'g'], 
#             #    ylims=[config['stateLowerLimit'][:nq], config['stateUpperLimit'][:nq]],
#                linestyle=['solid','solid', 'dotted', 'solid'])

# v_mea = get_v_(r.data['joint_positions'], r.data['joint_velocities'], pinrobot.model, pinrobot.model.getFrameId('contact'))
# s.plot_ee_vel( [v_mea], 
                  
#                ['CSSQP'], 
#                ['r', 'b'], 
#             #    ylims=[config['stateLowerLimit'][nq:], config['stateUpperLimit'][nq:]],
#                linestyle=['solid','solid'])

plt.show()