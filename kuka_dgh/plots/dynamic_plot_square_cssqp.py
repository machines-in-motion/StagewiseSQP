
from plot_utils import SimpleDataPlotter
from mim_data_utils import DataReader
from croco_mpc_utils.pinocchio_utils import *
import numpy as np
from demos import launch_utils

import matplotlib
import matplotlib.pyplot as plt 
matplotlib.use('GTK3Agg') 
from matplotlib.animation import FuncAnimation
import time


from robot_properties_kuka.config import IiwaConfig

iiwa_config = IiwaConfig()
pinrobot    = iiwa_config.buildRobotWrapper()
model       = pinrobot.model
data        = model.createData()
frameId     = model.getFrameId('contact')
nq = model.nq ; nv = model.nv
# Overwrite effort limit as in DGM
model.effortLimit = np.array([100, 100, 50, 50, 20, 10, 10])



# Load config file
SIM           = False
EXP_NAME      = 'square_cssqp' # <<<<<<<<<<<<< Choose experiment here (cf. launch_utils)
config        = launch_utils.load_config_file(EXP_NAME)


# Create data Plottger
s = SimpleDataPlotter(dt=1./config['ctrl_freq'])

if(SIM):
    data_path = '/home/skleff/data_sqp_paper_croc2/constrained/square/'
    data_name = 'square_cssqp_SIM_2023-10-20T17:25:45.051546_cssqp_best_filter=3' 
    
else:
    data_path = '/home/skleff/ws_croco2/workspace/src/StagewiseSQP/kuka_dgh/data/constrained/square/paper/'
    data_name = 'square_cssqp_REAL_2023-10-23T17:54:23.590863_cssqp'
    
r = DataReader(data_path+data_name+'.mds')
r1 = r
N = r.data['absolute_time'].shape[0]
print("Total number of control cycles = ", N)

# N_START = int(config['T_CIRCLE']*config['ctrl_freq'])
N_START = 0
print("N_START = ", N_START)


# Measured EE position 
p_mea = get_p_(r.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))
# p_des = get_p_(r.data['x_des'][:,:nq], pinrobot.model, pinrobot.model.getFrameId('contact'))
# Target EE position
target_position = np.zeros((N,3))
target_position[:,0] = r.data['target_position_x'][:,0]
target_position[:,1] = r.data['target_position_y'][:,0]
target_position[:,2] = r.data['target_position_z'][:,0]
# EE Bounds
ee_lb = r.data['lb'] 
ee_ub = r.data['ub']

# # Plot EE traj in the YZ plane 
# LINEWIDTH = 4
# fig, ax = plt.subplots(1, 1, figsize=(10.8,10.8)) 
# # Constraints
# ax.axvline(ee_lb[-1][1], color='k', linewidth=LINEWIDTH, linestyle='--', label='Constraint', alpha=0.6)
# ax.axvline(ee_ub[-1][1], color='k', linewidth=LINEWIDTH, linestyle='--', alpha=0.6)
# ax.axhline(ee_lb[-1][2], color='k', linewidth=LINEWIDTH, linestyle='--', alpha=0.6)
# ax.axhline(ee_ub[-1][2], color='k', linewidth=LINEWIDTH, linestyle='--', alpha=0.6)
# MAX = 5
# xmin = -0.47 ; xmax = +0.47
# ymin = +0.10 ; ymax = +1.00
# ax.set_xlim(xmin, xmax)
# ax.set_ylim(ymin, ymax)
# ax.axhspan(ee_ub[-1][2], MAX, -MAX, MAX, color='gray', alpha=0.2, lw=0)      # up
# ax.axhspan(-MAX, ee_lb[-1][2], -MAX, MAX, color='gray', alpha=0.2, lw=0)     # down
# ax.axvspan(ee_ub[-1][1], MAX, -MAX, MAX, color='gray', alpha=0.2, lw=0)      # up
# ax.axvspan(-MAX, ee_lb[-1][1], -MAX, MAX, color='gray', alpha=0.2, lw=0)     # down
# # Target 
# ax.plot(target_position[:,1], target_position[:,2], color='y', linewidth=LINEWIDTH, linestyle='-', label='Reference', alpha=1.) 
# # Measured
# ax.plot(p_mea[:,1], p_mea[:,2], color='b', linewidth=LINEWIDTH, label='Measured', alpha=0.5)
# # Axis label & ticks
# ax.set_ylabel('Z (m)', fontsize=26)
# ax.set_xlabel('Y (m)', fontsize=26)
# ax.tick_params(axis = 'y', labelsize=22)
# ax.tick_params(axis = 'x', labelsize=22)
# ax.grid(True) 
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.12, 0.885), prop={'size': 26}) 
# plt.show()

# save_path = save_path = os.path.join(SAVE_PATH, 'square_cssqp_plot.pdf')
# logger.warning("Saving figure to "+str(save_path))
# fig.savefig(save_path, bbox_inches="tight")


time_lin = np.linspace(0, (N-N_START)/config['ctrl_freq'], int(N-N_START))



# if(FILTER > 0):
#     print("FILTERING")
#     force_1 = analysis_utils.moving_average_filter(r1.data['contact_force_3d_measured'][N_START_1:N_END_1, 2:3].copy(), FILTER)
# else:
# force_1 = r1.data['contact_force_3d_measured'][N_START_1:N_END_1, 2:3]
# def compute_pos_error_traj(r):
    # p_mea = get_p_(r.data['joint_positions'][:N_TOT,:], pinrobot.model, pinrobot.model.getFrameId('contact'))
    # return (np.sqrt((p_mea[:, 0] - r.data['target_position_x'][:N_TOT,0])**2 + (p_mea[:, 1] - r.data['target_position_y'][:N_TOT,1])**2)).reshape((-1, 1))
# pos_error_1 = analysis_utils.moving_average_filter(compute_pos_error_traj(r1)[N_START_1:N_END_1].copy(), 100)



SPLIT = 10
time_lin_split = time_lin[N_START:N:SPLIT] 


p_mea_split = p_mea[::SPLIT]
p_ref_split = target_position #[::SPLIT]
# print(p_mea_split.shape)
# print(p_mea.shape)
color_list = ['b', 'g', 'r', 'y']


print("PLOTTING")


fig, ax = plt.subplots(1, 1, sharex='col', figsize=(68, 16))

fig.canvas.draw() 

# ax.plot(time_lin_split, target_force, color='k', linewidth=6, linestyle='--', label='Reference', alpha=0.5) 
ax.grid(linewidth=1)
ax.set_xlim(time_lin_split[0], time_lin_split[-1])
ax.set_ylim(25., 90.)
# ax.set_xlim(time_lin_split[0], time_lin_split[-1])
# ax.set_ylabel('Force (N)', fontsize=56)
ax.set_ylabel('Position error (m)', fontsize=56)
ax.set_xlabel('Time (s)', fontsize=56)

ax.tick_params(axis = 'y', labelsize=48)
ax.tick_params(axis = 'x', labelsize=48)
# ax.tick_params(labelbottom=False)  
ax.set_yticks([30, 50, 70, 90])

line1a = ax.plot(time_lin_split[0:1], p_mea_split[0:1,1], animated=True, color=color_list[0], linewidth=6, label='mea', alpha=0.8)
# line2a = ax.plot(time_lin_split[0:1], p_ref_split[0:1], animated=True, color=color_list[0], linewidth=6, label='ref', alpha=0.8)
# # Task phases
# PHASE_TIME = 6283
# ax.axvline(PHASE_TIME/1000., color='k', linewidth=4, linestyle='-', alpha=1.)
# ax.axvline(2*PHASE_TIME/1000., color='k', linewidth=4, linestyle='-', alpha=1.)
# ax.axvline(PHASE_TIME/1000., color='k', linewidth=4, linestyle='-', alpha=1.)
# ax.axvline(2*PHASE_TIME/1000., color='k', linewidth=4, linestyle='-', alpha=1.)
# fig.text(0.22, 0.90, 'Slow', transform=fig.transFigure, fontdict={'size':70})
# fig.text(0.5, 0.90, 'Medium',transform=fig.transFigure, fontdict={'size':70})
# fig.text(0.82, 0.90, 'Fast',transform=fig.transFigure, fontdict={'size':70})

line = line1a #, line2a]

ax.legend(loc="upper left", framealpha=0.95, fontsize=40) 

fig.align_ylabels()
plt.tight_layout(pad=1)


PPS = 100  # Point per second
T = (N-N_START)/config['ctrl_freq']
N_FRAMES = int(T * PPS)
SKIP = int(1000/PPS)





def init():
    """
    This init function defines the initial plot parameter
    """
    # Set initial parameter for the plot
    return line

def animate(t):
    """
    This function will be called periodically by FuncAnimation. Frame parameter will be passed on each call as a counter. 
    """
    
    mask1 = time_lin_split < t
    # print(line)
    # print(line[0])
    line[0].set_data(time_lin_split[mask1], p_mea_split[mask1,1])
    # line[1].set_data(time_lin_split[mask1], p_ref_split[mask1])

    # mask2 = time_lin_2 < t
    # line[4].set_data(time_lin_2[mask2], pos_error_2[mask2])
    # line[1].set_data(time_lin_2[mask2], force_2[mask2])

    # mask3 = time_lin_3 < t
    # line[2].set_data(time_lin_3[mask3], force_3[mask3])
    # line[5].set_data(time_lin_3[mask3], pos_error_3[mask3])

    return line

# Create FuncAnimation object and plt.show() to show the updated animation


t0 = time.time()


time_lin = np.linspace(0, T, N_FRAMES)
ani = FuncAnimation(fig, animate, frames=time_lin, repeat=False, interval = SKIP, init_func = init, blit=True)
folder = '/tmp/'
ani.save(folder + 'test.mp4') #, fps=PPS)


print("COMPUTE TIME = ", time.time() - t0)
# plt.show()

