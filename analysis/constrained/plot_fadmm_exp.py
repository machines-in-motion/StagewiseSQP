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

# Load experimental data

#     # Circle without constraint
# print("Extract data...")
# r1 = DataReader('/home/skleff/data_paper_fadmm/circle_no_cstr/no_constraint_1683299184.3249779.mds') 

#     # Cicle with joint 1 position constraint
# print("Extract data...")
# r2 = DataReader('/home/skleff/data_paper_fadmm/circle_jointpos_cstr/jointPos_constraint=0.05_1683299346.726773.mds')     

    # Circle with end-effector position constraint (D)                                  
print("Extract data...")
r3 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_D_1683299505.0607696.mds')  

#     # Circle with end-effector position constraint (square)    
# # r4 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_square_1683311293.0681663.mds')    
# # # r2 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_square_1683311101.474164.mds')    
# # # r2 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_square_1683319823.8800943.mds')    
# # # r2 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_square_1683321073.5147364.mds')    
# # # r2 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_square_fast_1683316565.3754733.mds')     # fast 
# # # r2 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_square_slow_1683318101.9747822.mds')     # slow 

# #     # Circle with end-effector position constraint (line)
# r5 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_line_1683299651.402261.mds')       
# # # r2 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_line_disturbance_1683299802.012319.mds')       

# #     # End-effector position constraint + disturbance (plane)
# r6 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_plane_cost_disturbance_1683301875.1998608.mds')    
# # # r2 = DataReader('/home/skleff/data_paper_fadmm/circle_endeff_cstr/endeff_constraint_plane_cost_disturbance_1683302232.4687898.mds')    

N = r3.data['tau'].shape[0]
# N = min(r1.data['tau'].shape[0], r2.data['tau'].shape[0])
# N = min(N, r3.data['tau'].shape[0])
# N = min(N, r4.data['tau'].shape[0])
# N = min(N, r5.data['tau'].shape[0])
# N = min(N, r6.data['tau'].shape[0])


# p_mea1 = get_p_(r1.data['joint_positions'][:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
# p_mea2 = get_p_(r2.data['joint_positions'][:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
p_mea3 = get_p_(r3.data['joint_positions'][:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
# p_mea4 = get_p_(r4.data['joint_positions'][:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
# p_mea5 = get_p_(r5.data['joint_positions'][:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
# p_mea6 = get_p_(r6.data['joint_positions'][:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
target_position = np.zeros((N, 3)) 
target_position[:,0] = r3.data['target_position_x'][:N,0]
target_position[:,1] = r3.data['target_position_y'][:N,0]
target_position[:,2] = r3.data['target_position_z'][:N,0]


# # Compute MSE ||y - yd||^2 + ||z - zd||^2 or constraint saturation / violation?
# err1 = np.sum(np.abs(p_mea1 - target_position)[2000:,1:]**2, axis=1)
# err2 = np.sum(np.abs(p_mea2 - target_position)[2000:,1:]**2, axis=1)
# print("MSE GNMS = ", np.average(err1))
# print("MSE FDDP = ", np.average(err2))

# Generate plot of number of iterations for each problem
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
 
# Plot end-effector trajectory (y,z) plane for ee constraints D and square, and line 
xdata = np.linspace(0, N*0.001, N+1) 
N_start = 2000


def plot_endeff_yz(pmea, name, plb=None, pub=None):
    fig0, ax0 = plt.subplots(1, 1, figsize=(19.2,10.8))
    # Target 
    # ax0.plot(p_mea1[N_start:N,1], p_mea1[N_start:N,2], color='r', linewidth=4, label='No constraint', alpha=0.5) 
    ax0.plot(target_position[N_start:N,1], target_position[N_start:N,2], color='y', linewidth=4, linestyle='--', label='Reference', alpha=1.) 
    # Measured
    ax0.plot(pmea[N_start:N,1], pmea[N_start:N,2], color='b', linewidth=4, label=name, alpha=0.5) 
    # Constraint
    ax0.axvline(0, color='k', linewidth=4, linestyle='--', label='Constraint')
    # Set axis and stuff
    ax0.set_xlim(-0.33, +0.33)
    ax0.set_ylim(0.15, 0.8)
    ax0.set_ylabel('Z (m)', fontsize=26)
    ax0.set_xlabel('Y (m)', fontsize=26)
    # ax0.set_ylim(-0.02, 1.02)
    ax0.tick_params(axis = 'y', labelsize=22)
    ax0.tick_params(axis = 'x', labelsize=22)
    ax0.grid(True) 
    # Legend 
    handles0, labels0 = ax0.get_legend_handles_labels()
    # fig0.legend(handles0, labels0, loc='lower right', bbox_to_anchor=(0.902, 0.1), prop={'size': 26}) 
    fig0.legend(handles0, labels0, loc='upper left', bbox_to_anchor=(0.12, 0.885), prop={'size': 26}) 
    # Save, show , clean
    fig0.savefig('/home/skleff/data_paper_fadmm/'+name+'_plot.pdf', bbox_inches="tight")
    return fig0, ax0


# fig1, ax1 = plot_endeff_yz(p_mea1, 'No constraint') 
# fig2, ax2 = plot_endeff_yz(p_mea2, 'Joint pos. constraint')
plb = np.zeros(p_mea3.shape); plb[:,:3] = np.array([-10.,-10, 0.])
pub = np.zeros(p_mea3.shape); pub[:,:3] = np.array([10.,10, 10.])
fig3, ax3 = plot_endeff_yz(p_mea3, 'End-effector constraint', plb, pub)




plt.show()
plt.close('all')

