
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


from mim_robots.robot_loader import load_pinocchio_wrapper

pinrobot    = load_pinocchio_wrapper('iiwa')
model       = pinrobot.model
data        = model.createData()
frameId     = model.getFrameId('contact')
nq = model.nq ; nv = model.nv
# Overwrite effort limit as in DGM
model.effortLimit = np.array([100, 100, 50, 50, 20, 10, 10])



# Load config file & create data plotter 
EXP_NAME  = 'line_cssqp'                                       # <<<<<<<<<<<<< Choose experiment here (cf. launch_utils)
config    = launch_utils.load_config_file(EXP_NAME)

s         = SimpleDataPlotter(dt=1./config['ctrl_freq'])
data_path = './data/constrained/line/'
# data_name = 'line_cssqp_REAL_2023-10-23T15:15:34.911408_cssqp_PUSH'
data_name = 'line_cssqp_REAL_2023-10-23T15:13:37.129836_cssqp' # <<<<<<<<<<<<< Choose data file here 

r         = DataReader(data_path+data_name+'.mds')
N         = r.data['absolute_time'].shape[0]
N_START   = int(config['T_CIRCLE']*config['ctrl_freq'])

logger.warning("Total number of control cycles = "+str(N))
logger.warning("N_START = "+str(N_START))


# Extract measured, target positions and constraint bounds
p_mea      = get_p_(r.data['joint_positions'][N_START:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
p_ref      = np.zeros((N-N_START,3))
p_ref[:,0] = r.data['target_position_x'][N_START:N,0]
p_ref[:,1] = r.data['target_position_y'][N_START:N,0]
p_ref[:,2] = r.data['target_position_z'][N_START:N,0]
p_lb      = r.data['lb'][-1] 
p_ub      = r.data['ub'][-1]



# Split trajectories 
SPLIT = 10
time_lin = np.linspace(0, (N-N_START)/config['ctrl_freq'], N-N_START)
time_lin_split = time_lin[::SPLIT] 
logger.warning("SPLIT                 : "+str(SPLIT))
logger.warning("time lin size         : "+str(time_lin.shape))
logger.warning("time lin size (split) : "+str(time_lin_split.shape))

p_mea_split = p_mea[::SPLIT]
p_ref_split = p_ref[::SPLIT]
p_lb_split = r.data['lb'][::SPLIT]
p_ub_split = r.data['ub'][::SPLIT]
cstr_split = r.data['constraint_norm'][N_START:N][::SPLIT]

# print(cstr_split)
color_list = ['r', 'y', 'k']

print("PLOTTING")


fig, ax = plt.subplots(2, 1, sharex='col', figsize=(16, 16))

fig.canvas.draw() 

from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt
def sci_format(x,lim):
    return '{:1.0e}'.format(x)
major_formatter = FuncFormatter(sci_format)


# Plot EE traj in the YZ plane 
LINEWIDTH = 6
ax[0].grid(linewidth=1)
ax[0].set_ylabel('Z (m)', fontsize=22)
ax[0].tick_params(axis = 'y', labelsize=22)
ax[0].set_ylim(0.1, 1.2)
ax[0].tick_params(labelbottom=False) 
# ax[0].set_yticks([0, 1, 2, 3, 4, 5])

# ax[1].set_ylim(0, 1e-5)s
ax[1].grid(linewidth=1)
ax[1].set_ylabel('Constraint norm', fontsize=22)
ax[1].tick_params(axis = 'y', labelsize=22)
ax[1].tick_params(axis = 'x', labelsize=22)
ax[1].set_xlabel('Time (s)', fontsize=22)
ax[1].set_xlim(time_lin_split[0], time_lin_split[-1])
ax[1].set_ylim(0, 1e-4)
ax[1].yaxis.set_major_formatter(major_formatter)

# Target z
ax[0].plot(time_lin_split, p_ref_split[:,2], color='y', linewidth=LINEWIDTH, linestyle='-', label='Reference', alpha=1.) 

# plt.show()


# Measured z
line_mea, = ax[0].plot(time_lin_split[0:1], p_mea_split[0:1,2], animated=True, color='b', linewidth=6, label='Measured', alpha=0.6)
# Constraint norm
line_cstr, = ax[1].plot(time_lin_split[0:1], cstr_split[0:1], animated=True, color='g', linewidth=6, label='Constraint norm', alpha=0.6)
# Stack animation objects
objects = [
           line_mea, 
           line_cstr,
            ] 
ax[0].legend(loc="upper left", framealpha=0.95, fontsize=26) 
ax[1].legend(loc="upper left", framealpha=0.95, fontsize=26) 
fig.align_ylabels()
plt.tight_layout(pad=1)
# ergreg


# Animation parameters
PPS      = 10  # Point per second
T        = (N-N_START)/config['ctrl_freq']
N_FRAMES = int(T * PPS)
SKIP     = int(1000/PPS)
logger.warning("PPS       : "+str(PPS))
logger.warning("T         : "+str(T))
logger.warning("N_FRAMES  : "+str(N_FRAMES))
logger.warning("SKIP      : "+str(SKIP))



def init():
    """
    This init function defines the initial plot parameter
    """
    # Set initial parameter for the plot
    return objects

def animate(t):
    """
    This function will be called periodically by FuncAnimation. Frame parameter will be passed on each call as a counter. 
    """
    
    mask = time_lin_split < t
    objects[0].set_data(time_lin_split[mask], p_mea_split[mask,2])
    objects[1].set_data(time_lin_split[mask], cstr_split[mask])
    return objects

# Create FuncAnimation object and plt.show() to show the updated animation


t0 = time.time()


time_lin = np.linspace(0, T, N_FRAMES)
ani = FuncAnimation(fig, animate, frames=time_lin, repeat=False, interval = SKIP, init_func = init, blit=True)
folder = '/tmp/'
ani.save(folder + 'line_cssqp_dynamic_plot.mp4') #, fps=PPS)


print("COMPUTE TIME = ", time.time() - t0)
# plt.show()

