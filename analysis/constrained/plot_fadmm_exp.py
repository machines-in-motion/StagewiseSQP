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
CONFIG_NAME = 'kuka_circle_fadmm'
CONFIG_PATH = "/home/skleff/misc_repos/gnms/demos/"+CONFIG_NAME+".yml"
config      = path_utils.load_yaml_file(CONFIG_PATH)

PLOTS = [
            'circle_no_cstr',
            'circle_joint_cstr',
            # 'circle_ee_cstr_D',
            # 'circle_ee_cstr_square',
            # 'circle_ee_cstr_line',
            # 'ee_cstr_plane'
        ]
Ns = []
rs = []

    
if('circle_no_cstr' in PLOTS):
    # Circle without constraint
    print("Extract circle_no_cstr data...")
    r1 = DataReader('/home/skleff/data_paper_fadmm/circle_no_cstr/no_constraint_1683299184.3249779.mds') 
    rs.append(r1)
    N = r1.data['tau'].shape[0] ; Ns.append(N)
    # p_mea1 = get_p_(r1.data['joint_positions'][:N], pinrobot.model, pinrobot.model.getFrameId('contact'))

if('circle_joint_cstr' in PLOTS):
    # Cicle with joint 1 position constraint
    print("Extract circle_joint_cstr data...") 
    r2 = DataReader('/home/skleff/data_paper_fadmm/circle_jointpos_cstr/jointPos_constraint=0.05_1683299346.726773.mds')   
    # r2 = DataReader('/home/skleff/data_paper_fadmm/circle_constraints_jointPos1.mds')   
    rs.append(r2) 
    N = r2.data['tau'].shape[0] ; Ns.append(N)
    # p_mea2 = get_p_(r2.data['joint_positions'][:N], pinrobot.model, pinrobot.model.getFrameId('contact'))

if('circle_ee_cstr_D' in PLOTS):
    # Circle with end-effector position constraint (D)                                  
    print("Extract circle_ee_cstr_D data...")
    r3 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_D_1683299505.0607696.mds')  
    rs.append(r3) 
    N = r3.data['tau'].shape[0] ; Ns.append(N)
    # p_mea3 = get_p_(r3.data['joint_positions'][:N], pinrobot.model, pinrobot.model.getFrameId('contact'))

if('circle_ee_cstr_square' in PLOTS):
    # Circle with end-effector position constraint (square) 
    print("Extract circle_ee_cstr_square data...")  
    r4 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_square_1683311293.0681663.mds')    
    rs.append(r4) 
    N = r4.data['tau'].shape[0] ; Ns.append(N)
    # p_mea4 = get_p_(r4.data['joint_positions'][:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
    # r2 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_square_1683311101.474164.mds')    
    # r2 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_square_1683319823.8800943.mds')    
    # r2 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_square_1683321073.5147364.mds')    
    # r2 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_square_fast_1683316565.3754733.mds')     # fast 
    # r2 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_square_slow_1683318101.9747822.mds')     # slow 

if('circle_ee_cstr_line' in PLOTS):
    # Circle with end-effector position constraint (line)
    print("Extract circle_ee_cstr_line data...")  
    r5 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_line_1683299651.402261.mds')      
    rs.append(r5) 
    N = r5.data['tau'].shape[0] ; Ns.append(N)
    # p_mea5 = get_p_(r5.data['joint_positions'][:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
    # r2 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_line_disturbance_1683299802.012319.mds')   

if('ee_cstr_plane' in PLOTS):
    # End-effector position constraint + disturbance (plane)
    print("Extract ee_cstr_plane data...")  
    r6 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_plane_cost_disturbance_1683301875.1998608.mds')    
    rs.append(r6) 
    N = r6.data['tau'].shape[0] ; Ns.append(N)
    # p_mea6 = get_p_(r6.data['joint_positions'][:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
    # r2 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_plane_cost_disturbance_1683302232.4687898.mds')    

N = min(Ns) 
N_start = 2000
target_position = np.zeros((N-N_start, 3)) 
xdata = np.linspace(0, (N-N_start)*0.001, N-N_start) 
target_position[:,0] = rs[0].data['target_position_x'][N_start:N,0]
target_position[:,1] = rs[0].data['target_position_y'][N_start:N,0]
target_position[:,2] = rs[0].data['target_position_z'][N_start:N,0]

# Generate plot of number of iterations for each problem
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
 
# Plot end-effector trajectory (y,z) plane 
def plot_endeff_yz(pmea, pref, label=None):
    fig0, ax0 = plt.subplots(1, 1, figsize=(10.8,10.8))
    # Target 
    ax0.plot(pref[:,1], pref[:,2], color='y', linewidth=4, linestyle='-', label='Reference', alpha=1.) 
    # Measured
    if(label is None):
        ax0.plot(pmea[:,1], pmea[:,2], color='b', linewidth=4, label='Measured', alpha=0.5)
    else: 
        ax0.plot(pmea[:,1], pmea[:,2], color='b', linewidth=4, label=label, alpha=0.5)
    # Axis label & ticks
    ax0.set_ylabel('Z (m)', fontsize=26)
    ax0.set_xlabel('Y (m)', fontsize=26)
    ax0.tick_params(axis = 'y', labelsize=22)
    ax0.tick_params(axis = 'x', labelsize=22)
    ax0.grid(True) 
    return fig0, ax0


 
# Plot end-effector trajectory (y,z) plane 
def plot_joint_traj(jmea, label):
    fig0, ax0 = plt.subplots(1, 1, figsize=(19.2,10.8))
    # Measured
    ax0.plot(xdata, jmea, color='b', linewidth=4, label=label, alpha=0.5) 
    # Axis label & ticks
    ax0.set_ylabel('Joint position $q_1$ (rad)', fontsize=26)
    ax0.set_xlabel('Time (s)', fontsize=26)
    ax0.tick_params(axis = 'y', labelsize=22)
    ax0.tick_params(axis = 'x', labelsize=22)
    ax0.grid(True) 
    return fig0, ax0


# if('circle_no_cstr' in PLOTS):
#     # Circle no constraint
#     p_mea = get_p_(r1.data['joint_positions'][N_start:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
#     fig0, ax0 = plot_endeff_yz(p_mea, target_position) 
#     ax0.set_xlim(-0.33, +0.33)
#     ax0.set_ylim(0.15, 0.8)
#     ax0.plot(p_mea[0,1], p_mea[0,2], 'ro', markersize=16)
#     ax0.text(0., 0.1, '$x_0$', fontdict={'size':26})
#     # handles, labels = ax0.get_legend_handles_labels()
#     # fig0.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.12, 0.885), prop={'size': 26}) 
#     # fig0.savefig('/home/skleff/data_paper_fadmm/no_cstr_circle_plot.pdf', bbox_inches="tight")
#     # Joint pos
#     jmea = r1.data['joint_positions'][N_start:N, 0]
#     fig1, ax1 = plot_joint_traj(jmea) 
#     ax1.set_ylim(-2., 2.)
#     # handles, labels = ax.get_legend_handles_labels()
#     # fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.12, 0.885), prop={'size': 26}) 
#     # fig.savefig('/home/skleff/data_paper_fadmm/no_cstr_q1_plot.pdf', bbox_inches="tight")


# Circle joint pos constraint
if('circle_joint_cstr' in PLOTS and 'circle_no_cstr' in PLOTS):
    # Circle
    p_mea1 = get_p_(r1.data['joint_positions'][N_start:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
    p_mea2 = get_p_(r2.data['joint_positions'][N_start:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
    fig_circle, ax_circle = plot_endeff_yz(p_mea2, target_position, "Constrained") 
    ax_circle.plot(p_mea1[:,1], p_mea1[:,2], color='g', linewidth=4, label='Unconstrained', alpha=0.5) 
    ax_circle.set_xlim(-0.33, +0.33)
    ax_circle.set_ylim(0.15, 0.8)
    ax_circle.plot(p_mea2[0,1], p_mea2[0,2], 'ro', markersize=16)
    ax_circle.text(0., 0.1, '$x_0$', fontdict={'size':26})
    handles, labels = ax_circle.get_legend_handles_labels()
    fig_circle.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.12, 0.885), prop={'size': 26}) 
    fig_circle.savefig('/home/skleff/data_paper_fadmm/jointpos_circle_plot.pdf', bbox_inches="tight")
    # Joint pos
    jmea1 = r1.data['joint_positions'][N_start:N, 0]
    jmea2 = r2.data['joint_positions'][N_start:N, 0]
    jlb = [-0.05]*(N-N_start) ; jub = [0.05]*(N-N_start)
    fig_q, ax_q = plot_joint_traj(jmea2, 'Constrained')
    # Constraint 
    ax_q.plot(xdata, jlb, color='k', linewidth=4, linestyle='--', label='Constraint', alpha=0.6)
    ax_q.plot(xdata, jub, color='k', linewidth=4, linestyle='--')
    MAX = 100
    ax_q.axhspan(jub[0], MAX, -MAX, MAX, color='gray', alpha=0.2, lw=0)      
    ax_q.axhspan(-MAX, jlb[0], -MAX, MAX, color='gray', alpha=0.2, lw=0)      
    ax_q.set_ylim(-0.75, 0.75)
    ax_q.set_xlim(0., 28)
    ax_q.plot(xdata, jmea1, color='g', linewidth=4, label='Unconstrained', alpha=0.5) 
    handles, labels = ax_q.get_legend_handles_labels()
    fig_q.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.12, 0.885), prop={'size': 26}) 
    fig_q.savefig('/home/skleff/data_paper_fadmm/jointpos_q1_plot.pdf', bbox_inches="tight")
    

# Circle D shape
if('circle_ee_cstr_D' in PLOTS):
    p_mea = get_p_(r3.data['joint_positions'][N_start:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
    fig, ax = plot_endeff_yz(p_mea, target_position) 
    ax.axvline(0., color='k', linewidth=4, linestyle='--', label='Constraint', alpha=0.6)
    ax.set_xlim(-0.33, +0.33)
    ax.set_ylim(0.15, 0.8)
    ax.axhspan(0.8, -0.5, 0.5, -0., color='gray', alpha=0.2, lw=0)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.12, 0.885), prop={'size': 26}) 
    fig.savefig('/home/skleff/data_paper_fadmm/circle_ee_cstr_D_plot.pdf', bbox_inches="tight")

# Circle square shape
if('circle_ee_cstr_square' in PLOTS):
    p_mea = get_p_(r4.data['joint_positions'][N_start:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
    fig, ax = plot_endeff_yz(p_mea, target_position) 
    plb = r4.data['lb_square']
    pub = r4.data['ub_square']
    ax.axvline(plb[0,1], color='k', linewidth=4, linestyle='--', label='Constraint', alpha=0.6)
    ax.axvline(pub[0,1], color='k', linewidth=4, linestyle='--', alpha=0.6)
    ax.axhline(plb[0,2], color='k', linewidth=4, linestyle='--', alpha=0.6)
    ax.axhline(pub[0,2], color='k', linewidth=4, linestyle='--', alpha=0.6)
    MAX = 100
    ax.axhspan(MAX, pub[0,2], 1., -0.5, color='gray', alpha=0.2, lw=0)       # up
    ax.axhspan(plb[0,2], -MAX, 1., -0.5, color='gray', alpha=0.2, lw=0)       # down
    ax.axhspan(MAX, -MAX, 0.5+pub[0,1], 1.1, color='gray', alpha=0.2, lw=0)  # right 
    ax.axhspan(MAX, -MAX, -MAX, 0.5+plb[0,1], color='gray', alpha=0.2, lw=0) # left
    ax.plot(p_mea[0,1], p_mea[0,2], 'ro', markersize=16)
    ax.text(0., 0.45, '$x_0$', fontdict={'size':26})
    ax.set_xlim(-0.5, +0.5)
    ax.set_ylim(0.18, 1.1)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.12, 0.885), prop={'size': 26}) 
    fig.savefig('/home/skleff/data_paper_fadmm/circle_ee_cstr_square_plot.pdf', bbox_inches="tight")



plt.show()
plt.close('all')

