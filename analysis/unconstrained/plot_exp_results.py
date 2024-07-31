from mim_data_utils import DataReader
from croco_mpc_utils.pinocchio_utils import *
import numpy as np
from robot_properties_kuka.config import IiwaConfig

# Generate plot of number of iterations for each problem
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import os
 
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
EXP_NAME      = 'circle_ssqp' # <<<<<<<<<<<<< Choose experiment here (cf. launch_utils)
config        = launch_utils.load_config_file(EXP_NAME, path_prefix=KUKA_DGH_PATH)


DATA_PATH = os.path.join(KUKA_DGH_PATH, 'data')
SAVE_PATH = '/tmp' # <<<<<<< EDIT SAVE PATH HERE

logger.info("Extract SQP data...")
r1 = DataReader(DATA_PATH+'/unconstrained/circle_ssqp_REAL_2023-10-31T16:45:47.050199_sqp.mds') 
# r1 = DataReader(DATA_PATH+'/unconstrained/circle_ssqp_REAL_2023-10-23T15:42:18.612802_sqp.mds') 
# r1 = DataReader(DATA_PATH+'/unconstrained/old/circle_ssqp_REAL_2023-10-19T16:41:31.062488_sqp.mds')  # old data

logger.info("Extract FDDP data...")
r2 = DataReader(DATA_PATH+'/unconstrained/circle_ssqp_REAL_2023-10-31T17:06:02.992743_fddp.mds') 
# r2 = DataReader(DATA_PATH+'/unconstrained/circle_ssqp_REAL_2023-10-23T15:45:07.463648_fddp.mds') 
# r2 = DataReader(DATA_PATH+'/unconstrained/old/circle_ssqp_REAL_2023-10-19T16:39:49.624312_fddp.mds') # old data
N       = min(r1.data['absolute_time'].shape[0], r2.data['absolute_time'].shape[0])
N_START = int(config['T_CIRCLE']*config['ctrl_freq'])
print("Total number of control cycles = ", N - N_START)
time_lin = np.linspace(0, (N-N_START)/config['ctrl_freq'], N-N_START)


# Plot the (total) cost over the whole trajectory ?
def compute_costs(r):
    state_cost_list       = []
    tau_cost_list         = []
    translation_cost_list = []
    total_cost_list       = []
    for index in range(N_START, N):
        state_mea = np.concatenate([r.data['joint_positions'][index,:], r.data['joint_velocities'][index,:]])
        tau_mea   = r.data['tau_ff'][index, :] + r.data['tau_gravity'][index,:]
        state_ref = np.array([0., 1.0471975511965976, 0., -1.1344640137963142, 0.2,  0.7853981633974483, 0, 0.,0.,0.,0.,0.,0.,0.])
        tau_ref   = r.data['tau_gravity'][index,:]
        p_mea     = get_p_(r.data['joint_positions'][index,:], model, frameId)
        p_ref     = np.zeros(3)
        p_ref[0] = r.data['target_position_x'][index,0]
        p_ref[1] = r.data['target_position_y'][index,0]
        p_ref[2] = r.data['target_position_z'][index,0]
        
        state_cost = 0.5 * config['stateRegWeight'] * (state_mea - state_ref).T @ np.diag(config['stateRegWeights'])**2 @ (state_mea - state_ref)
        state_cost_list.append(state_cost)

        tau_cost = 0.5 * config['ctrlRegGravWeight'] * (tau_mea - tau_ref).T @ np.diag(config['ctrlRegGravWeights'])**2 @ (tau_mea - tau_ref)
        tau_cost_list.append(tau_cost)

        translation_cost = 0.5 * config['frameTranslationWeight'] * (p_mea - p_ref).T @ np.diag(config['frameTranslationWeights'])**2 @ (p_mea - p_ref)
        translation_cost_list.append(translation_cost)
        
        total_cost = state_cost + tau_cost + translation_cost
        total_cost_list.append(total_cost)
    state_cost_       = np.array(state_cost_list).reshape(-1, 1)
    tau_cost_         = np.array(tau_cost_list).reshape(-1, 1)
    translation_cost_ = np.array(translation_cost_list).reshape(-1, 1)
    total_cost_       = np.array(total_cost_list).reshape(-1, 1)
    return state_cost_, tau_cost_, translation_cost_, total_cost_


# Plot the convergence ; KKT residual and number of iterations
fig, ax = plt.subplots(2, 1, sharex='col',  figsize=(13.8,10.8), constrained_layout=True)
LINEWIDTH = 6


from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt

def sci_format(x,lim):
    return '{:1.0e}'.format(x)

major_formatter = FuncFormatter(sci_format)
# ax.xaxis.set_major_formatter(major_formatter)


# KKT residual 
ax[0].plot(time_lin, r1.data['KKT'][N_START:N], label='SQP', linewidth=LINEWIDTH, color='b', alpha=0.5)
ax[0].plot(time_lin, r2.data['KKT'][N_START:N], label='FDDP',linewidth=LINEWIDTH,  color='g',  alpha=0.5)
ax[0].plot(time_lin, (N-N_START)*[config['solver_termination_tolerance']], label= 'Tolerance', linestyle='--', color='r', linewidth=LINEWIDTH, alpha=0.5)
ax[0].set_ylim(0, 5e-4)
ax[0].grid(linewidth=1)
ax[0].set_ylabel('KKT residual norm', fontsize=22)
ax[0].tick_params(axis = 'y', labelsize=22)
# ax[0].tick_params(axis = 'x', labelsize=22)
ax[0].tick_params(labelbottom=False) 
ax[0].yaxis.set_major_formatter(major_formatter)
# ax[0].set_yticks([0, 1, 2, 3, 4, 5])

# Number of iterations
ax[1].plot(time_lin, r1.data['ddp_iter'][N_START:N], label='SQP', linewidth=LINEWIDTH, color='b', alpha=0.5)
ax[1].plot(time_lin, r2.data['ddp_iter'][N_START:N], label='FDDP', linewidth=LINEWIDTH,  color='g', alpha=0.5)
ax[1].plot(time_lin, (N-N_START)*[config['maxiter']], label= 'Max. # iter.', linestyle='--', color='r', linewidth=LINEWIDTH, alpha=0.5)
ax[1].set_ylim(0, 6)
ax[1].set_yticks([0, 1, 2, 3, 4, 5])
ax[1].grid(linewidth=1)
ax[1].set_ylabel('Number of iterations', fontsize=22)
ax[1].tick_params(axis = 'y', labelsize=22)
ax[1].tick_params(axis = 'x', labelsize=22)
ax[1].set_xlabel('Time (s)', fontsize=22)
ax[1].set_xlim(time_lin[0], time_lin[-1])

# Cost
_,_,_, c1 = compute_costs(r1)
_,_,_, c2 = compute_costs(r2)
# print("Cumulative cost (log)      = ", np.sum(r.data['cost'][N_START:N]))
print("Cumulative cost of the MPC (SQP)  = ", np.sum(c1))
print("Cumulative cost of the MPC (FDDP) = ", np.sum(c2))
# ax[2].plot(time_lin, c1, label='SQP', linewidth=LINEWIDTH, color='b', alpha=0.8)
# ax[2].plot(time_lin, c2, label='FDDP', linewidth=LINEWIDTH,  color='g', alpha=0.8)
# # ax[2].plot(time_lin, (N-N_START)*[config['maxiter']], label= 'Max. # iter.', linestyle='--', color='r', linewidth=LINEWIDTH, alpha=0.5)
# # ax[1].set_ylim(0, 6)
# ax[2].set_xlabel('Time (s)', fontsize=26)
# ax[2].set_xlim(time_lin[0], time_lin[-1])


fig.align_ylabels()
handles, labels = ax[0].get_legend_handles_labels()
# ax[0].legend(loc="upper left", framealpha=0.95, fontsize=26) 
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.1, 1.), prop={'size': 26}) 
# plt.tight_layout(pad=1)
save_path = os.path.join(SAVE_PATH, 'circle_ssqp_vs_fddp_plot.pdf')
logger.warning("Saving figure to "+str(save_path))
fig.savefig(save_path, bbox_inches="tight")
plt.show() 


