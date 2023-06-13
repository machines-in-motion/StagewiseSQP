# from gnms.demos.plot_utils import SimpleDataPlotter
from mim_data_utils import DataReader
from core_mpc.pin_utils import *
import numpy as np
import pinocchio as pin
# import matplotlib.pyplot as plt 
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
CONFIG_NAME = 'kuka_circle_gnms'
CONFIG_PATH = "/home/skleff/misc_repos/gnms/demos/"+CONFIG_NAME+".yml"
config      = path_utils.load_yaml_file(CONFIG_PATH)

# Create data Plottger
# s = SimpleDataPlotter()

r1 = DataReader('/home/skleff/data_paper_fadmm/circle_GNMS.mds')
r2 = DataReader('/home/skleff/data_paper_fadmm/circle_FDDP.mds')

N = min(r1.data['tau'].shape[0], r1.data['tau'].shape[0])


# s.plot_joint_pos( [r.data['joint_positions'], r2.data['joint_positions']], 
#                    ['fddp', 'gnms'], 
#                    ['r', 'b'], 
#                    ylims=[model.lowerPositionLimit, model.upperPositionLimit] )
# s.plot_joint_vel( [r.data['joint_velocities'], r2.data['joint_velocities']], 
#                   ['fddp', 'gnms'], 
#                   ['r', 'b'], 
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


p_mea1 = get_p_(r1.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))
p_mea2 = get_p_(r2.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))
# p_des1 = get_p_(r1.data['x_des'][:,:nq], pinrobot.model, pinrobot.model.getFrameId('contact'))
target_position = np.zeros((N, 3)) #r.data['target_position'] #
target_position[:,0] = r1.data['target_position_x'][:,0]
target_position[:,1] = r1.data['target_position_y'][:,0]
target_position[:,2] = r1.data['target_position_z'][:,0]

v_mea1 = get_v_(r1.data['joint_positions'], r1.data['joint_velocities'], pinrobot.model, pinrobot.model.getFrameId('contact'))
v_mea2 = get_v_(r2.data['joint_positions'], r2.data['x_des'][:,nq:nq+nv], pinrobot.model, pinrobot.model.getFrameId('contact'))

# Generate plot of number of iterations for each problem
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
 
# x-axis : max number of iterations
xdata = np.linspace(0, N*0.001, N+1) 
N_start = 2000
fig0, ax0 = plt.subplots(1, 1, figsize=(19.2,10.8))
ax0.plot(p_mea1[N_start:N,0], p_mea1[N_start:N,1], color='r', linewidth=4, label='FDDP') 
ax0.plot(p_mea2[N_start:N,0], p_mea2[N_start:N,1], color='b', linewidth=4, label='SQP') 
# ax0.plot(target_position[N_start:N,0], target_position[N_start:N,1], color='y', linestyle='-.', linewidth=4, label='Reference') 
#     # Set axis and stuff
#     ax0.set_ylabel('Percentage of problems solved', fontsize=26)
#     ax0.set_xlabel('Max. number of iterations', fontsize=26)
#     ax0.set_ylim(-0.02, 1.02)
#     ax0.tick_params(axis = 'y', labelsize=22)
#     ax0.tick_params(axis = 'x', labelsize=22)
#     ax0.grid(True) 
#     # Legend 
#     handles0, labels0 = ax0.get_legend_handles_labels()
#     fig0.legend(handles0, labels0, loc='lower right', bbox_to_anchor=(0.902, 0.1), prop={'size': 26}) 
#     # Save, show , clean
# fig0.savefig('name.pdf', bbox_inches="tight")

plt.show()
plt.close('all')

