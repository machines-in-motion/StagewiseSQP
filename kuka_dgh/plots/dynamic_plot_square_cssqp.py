
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
EXP_NAME  = 'square_cssqp'                                       # <<<<<<<<<<<<< Choose experiment here (cf. launch_utils)
config    = launch_utils.load_config_file(EXP_NAME)

s         = SimpleDataPlotter(dt=1./config['ctrl_freq'])
data_path = './data/constrained/square/video/'
data_name = 'square_cssqp_REAL_2023-10-20T17:55:11.345520_cssqp' # <<<<<<<<<<<<< Choose data file here

r         = DataReader(data_path+data_name+'.mds')
N         = r.data['absolute_time'].shape[0]
N_START   = 0 #int(config['T_CIRCLE']*config['ctrl_freq'])

logger.warning("Total number of control cycles = "+str(N))
logger.warning("N_START = "+str(N_START))


# Extract measured, target positions and constraint bounds
p_mea      = get_p_(r.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))
p_ref      = np.zeros((N,3))
p_ref[:,0] = r.data['target_position_x'][:,0]
p_ref[:,1] = r.data['target_position_y'][:,0]
p_ref[:,2] = r.data['target_position_z'][:,0]
p_lb      = r.data['lb'][-1] 
p_ub      = r.data['ub'][-1]



# Split trajectories 
SPLIT = 10
time_lin = np.linspace(0, (N-N_START)/config['ctrl_freq'], int(N-N_START))
time_lin_split = time_lin[N_START:N:SPLIT] 
logger.warning("SPLIT                 : "+str(SPLIT))
logger.warning("time lin size         : "+str(time_lin.shape))
logger.warning("time lin size (split) : "+str(time_lin_split.shape))


p_mea_split = p_mea[::SPLIT]
p_ref_split = p_ref[::SPLIT]
p_lb_split = r.data['lb'][::SPLIT]
p_ub_split = r.data['ub'][::SPLIT]
color_list = ['r', 'y', 'k']

print("PLOTTING")


fig, ax = plt.subplots(1, 1, sharex='col', figsize=(16, 16))

fig.canvas.draw() 


# Plot EE traj in the YZ plane 
LINEWIDTH = 4
fig, ax = plt.subplots(1, 1, figsize=(10.8,10.8)) 
MAX_XY = 5
xmin = -0.47 ; xmax = +0.47
ymin = +0.10 ; ymax = +1.00
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_ylabel('Z (m)', fontsize=26)
ax.set_xlabel('Y (m)', fontsize=26)
ax.tick_params(axis = 'y', labelsize=22)
ax.tick_params(axis = 'x', labelsize=22)
# ax.tick_params(labelbottom=False) 
# ax.set_yticks([30, 50, 70, 90])
ax.grid(linewidth=1)

# Target 
ax.plot(p_ref_split[:,1], p_ref_split[:,2], color='y', linewidth=LINEWIDTH, linestyle='-', label='Reference', alpha=1.) 
# Measured
line_mea, = ax.plot(p_mea_split[0:1,1], p_mea_split[0:1,2], animated=True, color='b', linewidth=6, label='Measured', alpha=0.6)
# Constraints
line_lower  = ax.axhline(p_lb[2], xmin=0, xmax=0, color='k', animated=True, linewidth=LINEWIDTH, linestyle='--', label='Constraint', alpha=0.6) # lower
line_right  = ax.axvline(p_ub[1], ymin=0, ymax=0, color='k', animated=True, linewidth=LINEWIDTH, linestyle='--', alpha=0.6) # right
line_upper  = ax.axhline(p_ub[2], xmin=0, xmax=0, color='k', animated=True, linewidth=LINEWIDTH, linestyle='--', alpha=0.6) # upper
line_left   = ax.axvline(p_lb[1], ymin=0, ymax=0, color='k', animated=True, linewidth=LINEWIDTH, linestyle='--', alpha=0.6) # left
# These are not plt.lines.line2D but plt.patches.polygon objects
poly_lower = ax.axhspan(-MAX_XY, p_lb[2], -MAX_XY, -MAX_XY, animated=True, color='gray', alpha=0.2, lw=0)  # lower 
poly_right = ax.axvspan(p_ub[1], MAX_XY, -MAX_XY, -MAX_XY, animated=True, color='gray', alpha=0.2, lw=0)  # right
poly_upper = ax.axhspan(p_ub[2], MAX_XY, -MAX_XY, -MAX_XY, animated=True, color='gray', alpha=0.2, lw=0)   # upper
poly_left  = ax.axvspan(-MAX_XY, p_lb[1], -MAX_XY, -MAX_XY, animated=True, color='gray', alpha=0.2, lw=0)   # left
# Stack animation objects
objects = [
           line_mea, 
           line_lower, poly_lower, 
           line_right, poly_right, 
           line_upper, poly_upper,
           line_left, poly_left
            ] 
ax.legend(loc="upper left", framealpha=0.95, fontsize=26) 
fig.align_ylabels()
plt.tight_layout(pad=1)



# Animation parameters
PPS      = 10  # Point per second
T        = (N-N_START)/config['ctrl_freq']
N_FRAMES = int(T * PPS)
SKIP     = int(1000/PPS)
logger.warning("PPS       : "+str(PPS))
logger.warning("T         : "+str(T))
logger.warning("N_FRAMES  : "+str(N_FRAMES))
logger.warning("SKIP      : "+str(SKIP))
# Hard-coded times for constraints activation
T_LOWER = 11.40
T_RIGHT = 19.25
T_UPPER = 27.05
T_LEFT  = 35.02 


# wegweg
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
   
    objects[0].set_data(p_mea_split[mask,1], p_mea_split[mask,2])
    # line[1].set_data(p_lb_split[mask,1], p_lb_split[mask,2])

    if(t >= T_LOWER):
        objects[1].set_data(np.array([0, 1]), np.array([p_lb[2], p_lb[2]])) 
        xy = np.array([[-MAX_XY  , -MAX_XY    ],
                       [-MAX_XY  ,  p_lb[2]],
                       [ MAX_XY  ,  p_lb[2]],
                       [ MAX_XY  , -MAX_XY    ],
                       [-MAX_XY  , -MAX_XY    ]])
        objects[2].set_xy(xy)
    if(t >= T_RIGHT): 
        objects[3].set_data(np.array([p_ub[1], p_ub[1]]), np.array([0, 1]))
        xy = np.array([[ p_ub[1] , -MAX_XY ],
                       [ p_ub[1] ,  MAX_XY ],
                       [ MAX_XY     ,  MAX_XY ],
                       [ MAX_XY     , -MAX_XY ],
                       [ p_ub[1] , -MAX_XY ]])
        objects[4].set_xy(xy)
    if(t >= T_UPPER): 
        objects[5].set_data(np.array([0, 1]), np.array([p_ub[2], p_ub[2]])) 
        xy = np.array([[-MAX_XY , p_ub[2]],
                       [-MAX_XY , MAX_XY   ],
                       [ MAX_XY , MAX_XY   ],
                       [ MAX_XY , p_ub[2]],
                       [-MAX_XY , p_ub[2]]])
        objects[6].set_xy(xy)
    if(t >= T_LEFT): 
        objects[7].set_data(np.array([p_lb[1], p_lb[1]]), np.array([0, 1]))  
        xy = np.array([[-MAX_XY     , -MAX_XY],
                       [-MAX_XY     ,  MAX_XY],
                       [-p_lb[1] ,  MAX_XY],
                       [-p_lb[1] , -MAX_XY],
                       [-MAX_XY     , -MAX_XY]]) 
        objects[8].set_xy(xy)
        
    return objects

# Create FuncAnimation object and plt.show() to show the updated animation


t0 = time.time()


time_lin = np.linspace(0, T, N_FRAMES)
ani = FuncAnimation(fig, animate, frames=time_lin, repeat=False, interval = SKIP, init_func = init, blit=True)
folder = '/tmp/'
ani.save(folder + 'square_cssqp_dynamic_plot.mp4') #, fps=PPS)


print("COMPUTE TIME = ", time.time() - t0)
# plt.show()

