from mim_data_utils import DataReader
from croco_mpc_utils.pinocchio_utils import *
import numpy as np
from robot_properties_kuka.config import IiwaConfig

# Generate plot of number of iterations for each problem
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
 
 
KUKA_DGH_PATH   = os.path.join(os.path.dirname(__file__), '../../kuka_dgh')
os.sys.path.insert(1, str(KUKA_DGH_PATH))

from demos import launch_utils

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


iiwa_config = IiwaConfig()
pinrobot    = iiwa_config.buildRobotWrapper()
model       = pinrobot.model
data        = model.createData()
frameId     = model.getFrameId('contact')
nq = model.nq ; nv = model.nv

# Load config file
SIM           = False
EXP_NAME      = 'square_cssqp' # <<<<<<<<<<<<< Choose experiment here (cf. launch_utils)
config        = launch_utils.load_config_file(EXP_NAME, path_prefix=KUKA_DGH_PATH)

PLOTS = [
            'circle_ssqp',
            'circle_cssqp_joint',
            
            # 'circle_cssqp_ee',
            
            # 'square_cssqp',
            
            # 'line_cssqp',
            # 'plane_cssqp'
        ]
Ns = []
rs = []




DATA_PATH = os.path.join(KUKA_DGH_PATH, 'data')
SAVE_PATH = '/tmp' # <<<<<<< EDIT SAVE PATH HERE


# Circle without constraint (paper only)
if('circle_ssqp' in PLOTS):
    logger.info("Extract circle_no_cstr data...")
    r1 = DataReader(DATA_PATH+'/constrained/circle/joint_cstr/circle_cssqp_REAL_2023-10-23T16:53:14.202764_cssqp_UNCONSTRAINED.mds') 
    rs.append(r1)
    T_START         = 5.
    N = r1.data['absolute_time'].shape[0]
    Ns.append(N)
    logger.info("Total number of control cycles = "+str(N))

# Cicle with joint 1 position constraint
if('circle_cssqp_joint' in PLOTS):
    logger.info("Extract circle_cssqp_joint data...") 
    r2 = DataReader(DATA_PATH+'/constrained/circle/joint_cstr/circle_cssqp_REAL_2023-10-23T15:47:09.350083_cssqp.mds')   
    rs.append(r2) 
    T_START         = 5.
    N = r2.data['absolute_time'].shape[0]
    Ns.append(N)
    logger.info("Total number of control cycles = "+str(N))

# Cicle with EE half-plane constraint (paper only)
if('circle_cssqp_ee' in PLOTS):
    print("Extract circle_cssqp_ee data...")
    # r3 = DataReader(DATA_PATH+'/constrained/circle/ee_cstr/circle_cssqp_REAL_2023-10-23T16:23:11.176689_cssqp.mds')  
    r3 = DataReader(DATA_PATH+'/constrained/circle/ee_cstr/circle_cssqp_REAL_2023-10-23T17:22:23.308797_cssqp.mds')  
    rs.append(r3) 
    T_START         = 5.
    N = r3.data['absolute_time'].shape[0]
    Ns.append(N)
    logger.info("Total number of control cycles = "+str(N))

# Circle with end-effector position constraint (video + paper) 
if('square_cssqp' in PLOTS):
    print("Extract square_cssqp data...")  
    # r4 = DataReader(DATA_PATH+'/constrained/square/video/square_cssqp_REAL_2023-10-20T17:55:11.345520_cssqp.mds')  
    r4 = DataReader(DATA_PATH+'/constrained/square/paper/square_cssqp_REAL_2023-10-23T17:54:23.590863_cssqp.mds')  
    rs.append(r4) 
    T_START         = 33.
    N = r4.data['absolute_time'].shape[0]
    Ns.append(N)
    logger.info("Total number of control cycles = "+str(N))

# Circle with line constraint (video only)
if('line_cssqp' in PLOTS):
    print("Extract line_cssqp data...")  
    r5 = DataReader(DATA_PATH+'/constrained/line/line_cssqp_REAL_2023-10-23T15:13:37.129836_cssqp.mds')  
    # r5 = DataReader(data_path+'/constrained/line/line_cssqp_REAL_2023-10-23T15:15:34.911408_cssqp_PUSH.mds')  
    rs.append(r5) 
    
    N = r5.data['absolute_time'].shape[0]
    Ns.append(N)
    logger.info("Total number of control cycles = "+str(N))
    
# Plane constraint (video only)
if('plane_cssqp' in PLOTS):
    print("Extract plane_cssqp data...")  
    r6 = DataReader(DATA_PATH+'/constrained/plane/plane_cssqp_REAL_2023-10-20T18:39:32.202714_cssqp_PUSH_PLANE1.mds')  
    # r6 = DataReader(data_path+'/constrained/plane/plane_cssqp_REAL_2023-10-20T18:44:04.018319_cssqp_PUSH_PLANE2.mds')      
    rs.append(r6) 
    
    N = r6.data['absolute_time'].shape[0]
    Ns.append(N)
    logger.info("Total number of control cycles = "+str(N))
    
    
    
   
N               = min(Ns) 
N_START         = int(T_START * config['ctrl_freq'])
xdata           = np.linspace(0, (N-N_START)/config['ctrl_freq'], N-N_START) 


LINEWIDTH = 8

# Plot end-effector trajectory (y,z) plane 
def plot_endeff_yz(fig0, ax0, pmea, pref, label=None):
    # Target 
    ax0.plot(pref[:,1], pref[:,2], color='y', linewidth=LINEWIDTH, linestyle='-', label='Reference', alpha=1.) 
    # Measured
    if(label is None):
        ax0.plot(pmea[:,1], pmea[:,2], color='b', linewidth=LINEWIDTH, label='Measured', alpha=0.5)
    else: 
        ax0.plot(pmea[:,1], pmea[:,2], color='b', linewidth=LINEWIDTH, label=label, alpha=0.5)
    # Axis label & ticks
    ax0.set_ylabel('Z (m)', fontsize=26)
    ax0.set_xlabel('Y (m)', fontsize=26)
    ax0.tick_params(axis = 'y', labelsize=22)
    ax0.tick_params(axis = 'x', labelsize=22)
    ax0.grid(True) 
    return fig0, ax0

 
# Plot end-effector trajectory (y,z) plane 
def plot_joint_traj(fig0, ax0, jmea, label):
    # Measured
    ax0.plot(xdata, jmea, color='b', linewidth=LINEWIDTH, label=label, alpha=0.5) 
    # Axis label & ticks
    ax0.set_ylabel('Joint position $q_1$ (rad)', fontsize=26)
    ax0.set_xlabel('Time (s)', fontsize=26)
    ax0.tick_params(axis = 'y', labelsize=22)
    ax0.tick_params(axis = 'x', labelsize=22)
    ax0.grid(True) 
    return fig0, ax0

# Circle joint pos constraint
if('circle_cssqp_joint' in PLOTS and 'circle_ssqp' in PLOTS):
    # Circle
    target_position = np.zeros((N-N_START, 3)) 
    target_position[:,0] = rs[0].data['target_position_x'][N_START:N,0]
    target_position[:,1] = rs[0].data['target_position_y'][N_START:N,0]
    target_position[:,2] = rs[0].data['target_position_z'][N_START:N,0]
    p_mea1 = get_p_(r1.data['joint_positions'][N_START:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
    p_mea2 = get_p_(r2.data['joint_positions'][N_START:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
    fig_circle, ax_circle = plt.subplots(1, 1, figsize=(10.8,10.8)) 
    plot_endeff_yz(fig_circle, ax_circle, p_mea2, target_position, "Constrained") 
    ax_circle.plot(p_mea1[:,1], p_mea1[:,2], color='g', linewidth=LINEWIDTH, label='Unconstrained', alpha=0.5) 
    ax_circle.set_xlim(-0.33, +0.33)
    ax_circle.set_ylim(0.12, 0.8)
    ax_circle.plot(p_mea2[0,1], p_mea2[0,2], 'ro', markersize=16)
    # ax_circle.text(0., 0.2, '$x_0$', fontdict={'size':26})
    handles, labels = ax_circle.get_legend_handles_labels()
    fig_circle.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.12, 0.885), prop={'size': 26}) 
    save_path = save_path = os.path.join(SAVE_PATH, 'circle_cssqp_joint_plot2.pdf')
    logger.warning("Saving figure to "+str(save_path))
    fig_circle.savefig(save_path, bbox_inches="tight")
    # Joint pos
    jmea1 = r1.data['joint_positions'][N_START:N, 0]
    jmea2 = r2.data['joint_positions'][N_START:N, 0]
    jlb = [-0.05]*(N-N_START) ; jub = [0.05]*(N-N_START)
    fig_q, ax_q = plt.subplots(1, 1, figsize=(19.2,10.8))
    # Constraint 
    ax_q.plot(xdata, jlb, color='k', linewidth=LINEWIDTH, linestyle='--', label='Constraint', alpha=0.6)
    ax_q.plot(xdata, jub, color='k', linewidth=LINEWIDTH, linestyle='--', alpha=0.6)
    MAX = 100
    ax_q.axhspan(jub[0], MAX, -MAX, MAX, color='gray', alpha=0.2, lw=0)      
    ax_q.axhspan(-MAX, jlb[0], -MAX, MAX, color='gray', alpha=0.2, lw=0)    
    plot_joint_traj(fig_q, ax_q, jmea2, 'Constrained')  
    ax_q.set_ylim(-0.75, 0.75)
    ax_q.set_xlim(0., (N-N_START)/config['ctrl_freq'])
    ax_q.plot(xdata, jmea1, color='g', linewidth=LINEWIDTH, label='Unconstrained', alpha=0.5) 
    handles, labels = ax_q.get_legend_handles_labels()
    fig_q.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.12, 0.885), prop={'size': 26}) 
    save_path = os.path.join(SAVE_PATH, 'circle_cssqp_joint_plot.pdf')
    logger.warning("Saving figure to "+str(save_path))
    fig_q.savefig(save_path, bbox_inches="tight")
    

# Circle D shape
if('circle_cssqp_ee' in PLOTS):
    target_position = np.zeros((N-N_START, 3)) 
    target_position[:,0] = rs[0].data['target_position_x'][N_START:N,0]
    target_position[:,1] = rs[0].data['target_position_y'][N_START:N,0]
    target_position[:,2] = rs[0].data['target_position_z'][N_START:N,0]
    p_mea = get_p_(r3.data['joint_positions'][N_START:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
    fig, ax = plt.subplots(1, 1, figsize=(10.8,10.8)) 
    ax.axvline(0., color='k', linewidth=LINEWIDTH, linestyle='--', label='Constraint', alpha=0.6)
    ax.set_xlim(-0.33, +0.33)
    ax.set_ylim(0.12, 0.8)
    ax.axhspan(0.8, -0.5, 0.5, -0., color='gray', alpha=0.2, lw=0)
    plot_endeff_yz(fig, ax, p_mea, target_position) 
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.12, 0.885), prop={'size': 26}) 
    save_path = save_path = os.path.join(SAVE_PATH, 'circle_cssqp_ee_plot.pdf')
    logger.warning("Saving figure to "+str(save_path))
    fig.savefig(save_path, bbox_inches="tight")

# Circle square shape
if('square_cssqp' in PLOTS):
    target_position = np.zeros((N-N_START, 3)) 
    target_position[:,0] = r4.data['target_position_x'][N_START:N,0]
    target_position[:,1] = r4.data['target_position_y'][N_START:N,0]
    target_position[:,2] = r4.data['target_position_z'][N_START:N,0]
    p_mea = get_p_(r4.data['joint_positions'][N_START:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
    fig, ax = plt.subplots(1, 1, figsize=(10.8,10.8)) 
    
    plb = r4.data['lb'][-1]
    pub = r4.data['ub'][-1]
    ax.axvline(plb[1], color='k', linewidth=LINEWIDTH, linestyle='--', label='Constraint', alpha=0.6)
    ax.axvline(pub[1], color='k', linewidth=LINEWIDTH, linestyle='--', alpha=0.6)
    ax.axhline(plb[2], color='k', linewidth=LINEWIDTH, linestyle='--', alpha=0.6)
    ax.axhline(pub[2], color='k', linewidth=LINEWIDTH, linestyle='--', alpha=0.6)
    MAX = 5
    xmin = -0.47 ; xmax = +0.47
    ymin = +0.10 ; ymax = +1.00
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axhspan(pub[2], MAX, -MAX, MAX, color='gray', alpha=0.2, lw=0)      # up
    ax.axhspan(-MAX, plb[2], -MAX, MAX, color='gray', alpha=0.2, lw=0)     # down
    ax.axvspan(pub[1], MAX, -MAX, MAX, color='gray', alpha=0.2, lw=0)      # up
    ax.axvspan(-MAX, plb[1], -MAX, MAX, color='gray', alpha=0.2, lw=0)     # down
    plot_endeff_yz(fig, ax, p_mea, target_position) 
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.12, 0.885), prop={'size': 26}) 
    save_path = save_path = os.path.join(SAVE_PATH, 'square_cssqp_plot.pdf')
    logger.warning("Saving figure to "+str(save_path))
    fig.savefig(save_path, bbox_inches="tight")



plt.show()
plt.close('all')

