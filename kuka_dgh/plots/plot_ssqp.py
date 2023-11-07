from plot_utils import SimpleDataPlotter
from mim_data_utils import DataReader
from croco_mpc_utils.pinocchio_utils import *
import numpy as np
import matplotlib.pyplot as plt 
from demos import launch_utils


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
EXP_NAME      = 'circle_ssqp' # <<<<<<<<<<<<< Choose experiment here (cf. launch_utils)
config        = launch_utils.load_config_file(EXP_NAME)


# Create data Plottger
s = SimpleDataPlotter()

if(SIM):
    data_path = '/tmp/'
    data_name = 'circle_ssqp_SIM_2023-10-18T14:17:41.165916_sqp' 
    
else:
    data_path = 'data/unconstrained/new/'
    data_name = 'circle_ssqp_REAL_2023-10-31T17:06:02.992743_fddp' 
    # data_name = 'circle_ssqp_REAL_2023-10-31T16:45:47.050199_sqp' 

r       = DataReader(data_path+data_name+'.mds')
N       = r.data['absolute_time'].shape[0]
print("Total number of control cycles = ", N)
time_lin = np.linspace(0, N/config['ctrl_freq'], N)


fig, ax = plt.subplots(4, 1, sharex='col') 
ax[0].plot(r.data['KKT'], label='KKT residual')
ax[0].plot(N*[config['solver_termination_tolerance']], label= 'KKT residual tolerance', color = 'r')
ax[1].plot(r.data['ddp_iter'], label='# solver iterations')
ax[2].plot(r.data['t_child']*1000, label='OCP solve time')
ax[2].plot(N*[1000./config['ctrl_freq']], label= 'dt_MPC', color='r')
ax[3].plot((r.data['timing_control'])* 1000, label='Control cycle time')
ax[3].plot(N*[1000./config['ctrl_freq']], label= 'dt_MPC', color='r')
for i in range(4):
    ax[i].grid()
    ax[i].legend()


s.plot_joint_pos( [r.data['joint_positions'], 
                   r.data['x_des'][:,:nq]], 
                   ['mea', 
                    'pred'], 
                   ['r', 
                    'b'],
                   ylims=[model.lowerPositionLimit, model.upperPositionLimit] )
# s.plot_joint_vel( [r.data['joint_velocities'], r.data['x_des'][:,nq:nq+nv]], # r.data['x'][:,nq:nq+nv], r.data['x1'][:,nq:nq+nv]],
#                   ['mea', 'pred'], # 'pred0', 'pred1'], 
#                   ['r', 'b'], #[0.2, 0.2, 0.2, 0.5], 'b', 'g']) 
#                   ylims=[-model.velocityLimit, +model.velocityLimit] )

# For SIM robot only
if(SIM):
    s.plot_joint_tau( [r.data['tau'], 
                       r.data['tau_ff']],
                    #    r.data['tau_riccati'], 
                    #    r.data['tau_gravity']], 
                      ['total', 
                       'ff', 
                       'riccati', 
                       'gravity'], 
                      ['r', 
                       'g', 
                       'b', 
                       [0.2, 0.2, 0.2, 0.5]],
                      ylims=[-model.effortLimit, +model.effortLimit] )
# For REAL robot only !! DEFINITIVE FORMULA !!
else:
    # Our self.tau was subtracted gravity, so we add it again
    # joint_torques_measured DOES include the gravity torque from KUKA
    # There is a sign mismatch in the axis so we use a minus sign
    s.plot_joint_tau( [-r.data['joint_cmd_torques'], 
                       r.data['joint_torques_measured'], 
                       r.data['tau'] + r.data['tau_gravity'], 
                       r.data['tau_ff'] + r.data['tau_gravity']], 
                  ['-cmd (FRI)', 
                   'Measured', 
                   'Desired (sent to robot) [+g(q)]', 
                   'tau_ff (OCP solution) [+g(q)]', 
                   'Measured - EXT'], 
                  ['k', 'r', 'b', 'g', 'y'],
                  ylims=[-model.effortLimit, +model.effortLimit],
                  linestyle=['dotted', 'solid', 'solid', 'solid', 'solid'])

p_mea = get_p_(r.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))
p_des = get_p_(r.data['x_des'][:,:nq], pinrobot.model, pinrobot.model.getFrameId('contact'))

if(EXP_NAME == 'reach_ssqp'):
    target_position = r.data['target_position'] #np.zeros((N,3))
else:
    target_position = np.zeros((N,3))
    target_position[:,0] = r.data['target_position_x'][:,0]
    target_position[:,1] = r.data['target_position_y'][:,0]
    target_position[:,2] = r.data['target_position_z'][:,0]
s.plot_ee_pos( [p_mea, 
                target_position],  
               ['mea', 
                'ref (position cost)'], 
               ['r',  
                'k', 
                'g'], 
               linestyle=['solid', 'dotted', 'solid'])




# Compute the total cost of the experiment 
state_cost_list       = []
tau_cost_list         = []
translation_cost_list = []
total_cost_list       = []
N_START = int(config['T_CIRCLE']*config['ctrl_freq'])
for index in range(N_START, N):
    state_mea = np.concatenate([r.data['joint_positions'][index,:], r.data['joint_velocities'][index,:]])
    tau_mea   = r.data['tau_ff'][index, :]
    state_ref = np.array([0., 1.0471975511965976, 0., -1.1344640137963142, 0.2,  0.7853981633974483, 0, 0.,0.,0.,0.,0.,0.,0.])
    tau_ref   = r.data['tau_gravity'][index,:]
    
    state_cost = 0.5 * config['stateRegWeight'] * (state_mea - state_ref).T @ np.diag(config['stateRegWeights'])**2 @ (state_mea - state_ref)
    state_cost_list.append(state_cost)

    tau_cost = 0.5 * config['ctrlRegGravWeight'] * (tau_mea - tau_ref).T @ np.diag(config['ctrlRegGravWeights'])**2 @ (tau_mea - tau_ref)
    tau_cost_list.append(tau_cost)

    translation_cost = 0.5 * config['frameTranslationWeight'] * (p_mea[index, :] - target_position[index, :]).T @ np.diag(config['frameTranslationWeights'])**2 @ (p_mea[index, :] - target_position[index, :])
    translation_cost_list.append(translation_cost)
    
    total_cost = state_cost + tau_cost + translation_cost
    total_cost_list.append(total_cost)
state_cost_       = np.array(state_cost_list).reshape(-1, 1)
tau_cost_         = np.array(tau_cost_list).reshape(-1, 1)
translation_cost_ = np.array(translation_cost_list).reshape(-1, 1)
total_cost_       = np.array(total_cost_list).reshape(-1, 1)



ANIMATION = True
LINEWIDTH = 6
ALPHA = 0.8
 

print("PLOTTING")

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex='col', figsize=(55, 13.5))
 

ax1.grid(linewidth=1) 
ax2.grid(linewidth=1) 
ax3.grid(linewidth=1) 
ax4.grid(linewidth=1) 

ax1.set_xlim(time_lin[0], time_lin[-1])
ax2.set_xlim(time_lin[0], time_lin[-1])
ax3.set_xlim(time_lin[0], time_lin[-1])
ax4.set_xlim(time_lin[0], time_lin[-1])

# ax1.set_ylim(0., 0.7)
# ax2.set_ylim(0., 1.1)
# ax3.set_ylim(0., 1.6)
   
ax1.set_ylabel('State cost ', fontsize=20)
ax2.set_ylabel('Contrl cost ', fontsize=20)
ax3.set_ylabel('Translation cost ' , fontsize=20)
ax4.set_ylabel('Total cost' , fontsize=20)
ax4.set_xlabel('Time (s)', fontsize=20)

ax1.tick_params(axis = 'y', labelsize=38)
ax2.tick_params(axis = 'y', labelsize=38)
ax3.tick_params(axis = 'y', labelsize=38)
ax4.tick_params(axis = 'y', labelsize=38)
ax4.tick_params(axis = 'x', labelsize=38)

ax1.tick_params(labelbottom=False)  
ax2.tick_params(labelbottom=False)  
ax3.tick_params(labelbottom=False)  

time_lin2 = time_lin[N_START:N]
ax1.plot(time_lin2, state_cost_,  linewidth=LINEWIDTH, label='Sate cost')
ax2.plot(time_lin2, tau_cost_, linewidth=LINEWIDTH, label='Ctrl cost')
ax3.plot(time_lin2, translation_cost_, linewidth=LINEWIDTH, label='Translation cost')
ax4.plot(time_lin2, total_cost_, linewidth=LINEWIDTH, label='Total cost (offline computed)')
fig.legend()
# print("Cumulative cost (log)      = ", np.sum(r.data['cost'][N_START:N]))
print("Cumulative cost of the MPC = ", np.sum(total_cost_))
plt.show()






