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
CONFIG_PATH = "demos/"+CONFIG_NAME+".yml"
config      = path_utils.load_yaml_file(CONFIG_PATH)


# Load data 
SIM = False 


# Create data Plottger
s = SimpleDataPlotter()


if(SIM):
    # r = DataReader('/home/skleff/Desktop/circle_PROXQP.mds')
    # r1 = DataReader('/home/skleff/Desktop/circle_CSSQP.mds')
    # r1 = DataReader('/tmp/kuka_circle_sim_PROXQP.mds')
    r1 = DataReader('/tmp/kuka_square_sim_square_constraintEEy.mds') #r2
    r2 = r1
    # r2 = DataReader('/tmp/kuka_circle_sim_PROXQP_warm_start_y=False_reset_rho=Falsesqp3.mds') #r2
    # r2 = r1 #DataReader('/tmp/kuka_circle_sim_CSSQP_warm_start_y=True_reset_rho=False.mds')
    r3 = r1 #DataReader('/tmp/kuka_circle_sim_CSSQP_warm_start_y=True_reset_rho=True.mds')
    r4 = r1 #DataReader('/tmp/kuka_circle_sim_CSSQP_warm_start_y=False_reset_rho=Truesqp3.mds')
    # r2 = DataReader('/tmp/kuka_circle_sim_CSSQP_NO_CONSTRAINT.mds')
else:
    # r = DataReader('/home/skleff/Desktop/circle_PROXQP.mds')
    # r1 = DataReader('/home/skleff/Desktop/circle_CSSQP.mds')
    # r1 = DataReader('/tmp/kuka_circle_real_=PROXQP.mds')

    # r1 = DataReader('/tmp/kuka_circle_real_=CSSQP_NO_CONSTRAINT.mds')  # baseline
    # r1 = DataReader('/tmp/kuka_circle_real_CSSQP_no_constraint.mds')  
    # r1 = DataReader('/home/skleff/data_paper_CSSQP/circle_endeff_cstr/endeff_constraint_square_1683311293.0681663.mds')  
    r1 = DataReader('/home/skleff/data_paper_CSSQP/circle_no_cstr/no_constraint_1683299184.3249779.mds')  
    # r1 = DataReader('/tmp/kuka_circle_real_PROXQP_warm_start_y=False_reset_rho=False_allJoints.mds')  # current best
    r2 = r1
    r3 = r1
N = r1.data['tau'].shape[0]


fig, ax = plt.subplots(6, 1, sharex='col') 
ax[0].plot(r1.data['qp_iters'], label='qp_iters (1)')
ax[0].plot(r2.data['qp_iters'], label='qp_iters (2)')
ax[0].plot(r3.data['qp_iters'], label='qp_iters (3)')
# ax[0].plot(r4.data['qp_iters'], label='qp_iters (4)')

ax[1].plot(r1.data['gap_norm'], label='gap_norm (1)')
ax[1].plot(r2.data['gap_norm'], label='gap_norm (2)')
ax[1].plot(r3.data['gap_norm'], label='gap_norm (3)')
# ax[1].plot(r4.data['gap_norm'], label='gap_norm (4)')

ax[2].plot(r1.data['constraint_norm'], label='constraint_norm (1)')
ax[2].plot(r2.data['constraint_norm'], label='constraint_norm (2)')
ax[2].plot(r3.data['constraint_norm'], label='constraint_norm (3)')
# ax[2].plot(r4.data['constraint_norm'], label='constraint_norm (4)')

ax[3].plot(r1.data['kkt_norm'], label='KKT norm (1)')
ax[3].plot(r2.data['kkt_norm'], label='KKT norm (2)')
ax[3].plot(r3.data['kkt_norm'], label='KKT norm (3)')
# ax[3].plot(r4.data['kkt_norm'], label='KKT norm (4)')

ax[4].plot(r1.data['t_child'], label='CSSQP solve (1)')
ax[4].plot(r2.data['t_child'], label='CSSQP solve (3)')
ax[4].plot(r3.data['t_child'], label='CSSQP solve (3)')
# ax[4].plot(r4.data['t_child'], label='CSSQP solve (4)')
ax[4].plot(N*[1./config['plan_freq']], label= 'mpc')

ax[5].plot(r1.data['t_run'], label='CSSQP cycle (1)')
ax[5].plot(r2.data['t_run'], label='CSSQP cycle (3)')
ax[5].plot(r3.data['t_run'], label='CSSQP cycle (4)')
ax[5].plot(N*[1./config['plan_freq']], label= 'mpc')
fig.legend() 


# Limits
xlb = config['stateLowerLimit']
xub = config['stateUpperLimit']

qlb = np.array([xlb[:nq]]*N) ; qub = np.array([xub[:nq]]*N)
vlb = np.array([xlb[nq:]]*N) ; vub = np.array([xub[nq:]]*N)

eps = 0.05

s.plot_joint_pos( [r2.data['joint_positions'], r1.data['joint_positions'], qlb, qub], 
                   ['Data 2', 'Data 1', 'lb', 'ub'], 
                   ['r', 'b', 'k', 'k'], 
                   linestyle=['solid','solid', 'dotted', 'dotted'],
                   ylims=[model.lowerPositionLimit-eps, model.upperPositionLimit+eps] )
s.plot_joint_vel( [r1.data['joint_velocities'], 
                   r2.data['joint_velocities'],
                   r3.data['joint_velocities'],
                #    r4.data['joint_velocities'], 
                   vlb, vub], 
                  ['Data 1', 'Data 2', 'Data 3', 'lb', 'ub'], 
                  ['r', 'b', 'g', 'k', 'k'], 
                  linestyle=['solid','solid', 'solid', 'dotted', 'dotted'],
                  ylims=[-model.velocityLimit-eps, +model.velocityLimit+eps] )

ulb = -np.array([config['ctrlLimit']]*N) 
uub = np.array([config['ctrlLimit']]*N) 
# For SIM robot only
if(SIM):
    s.plot_joint_tau( [r2.data['tau'], r1.data['tau'], ulb, uub],
                      ['Data 2', 'Data 1', 'lb', 'ub'], 
                      ['r', 'b', 'k', 'k'],
                      linestyle=['solid','solid', 'dotted', 'dotted'],
                      ylims=[-model.effortLimit-100*eps, +model.effortLimit+100*eps] )
# For REAL robot only !! DEFINITIVE FORMULA !!
else:
    # Our self.tau was subtracted gravity, so we add it again
    # joint_torques_measured DOES include the gravity torque from KUKA
    # There is a sign mismatch in the axis so we use a minus sign
    s.plot_joint_tau( [r1.data['joint_torques_measured'], r1.data['tau'] + r1.data['tau_gravity'],
                       r2.data['joint_torques_measured'], r2.data['tau'] + r2.data['tau_gravity'], 
                       ulb, uub], 
                  ['Data 1 (mea)', 'Data 1 (des + g)', 'Data 2 (mea)', 'Data 2 (des + g)', 'lb', 'ub'], 
                  ['b', 'b', 'r', 'r',  'k', 'k'],
                  linestyle=['solid','dotted', 'solid', 'dotted', 'dotted', 'dotted'],
                  ylims=[-model.effortLimit-10*eps, +model.effortLimit+10*eps] )
    # s.plot_joint_tau([-r.data['joint_cmd_torques'],  
    #                   r.data['joint_torques_measured'],  
    #                   r.data['tau']], labels=['cmd', 'mea', 'sent'], 
    #                   colors=['k', 'g', 'b'])



# p_mea = get_p_(r.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))
p_mea1 = get_p_(r1.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))
p_mea2 = get_p_(r2.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))
p_mea3 = get_p_(r3.data['joint_positions'], pinrobot.model, pinrobot.model.getFrameId('contact'))
p_des = get_p_(r1.data['x_des'][:,:nq], pinrobot.model, pinrobot.model.getFrameId('contact'))

plb = 0.*p_mea1 #r1.data['lb_square']
pub = 0.*p_mea1 #r1.data['ub_square']
print()
# pub = np.array([np.inf, r1.data['center_y'] + r1.data['radius2'], r1.data['center_z'] + r1.data['radius2']]) 
target_position = np.zeros((N, 3)) #r.data['target_position'] #
target_position[:,0] = r1.data['target_position_x'][:,0]
target_position[:,1] = r1.data['target_position_y'][:,0]
target_position[:,2] = r1.data['target_position_z'][:,0]
s.plot_ee_pos( [p_mea1,
                p_mea2,
                p_mea3,
                target_position,
                plb, pub,],  
               ['Data 1', 'Data 2','Data 3', 'Reference', 'lb', 'ub'], 
               ['b', 'r', 'g', 'y', 'k', 'k'], 
               linestyle=['solid','solid', 'solid', 'dotted', 'dotted', 'dotted'])

plt.figure()
plt.plot(p_mea1[:,1], p_mea1[:,2])
plt.plot(p_des[:,1], p_des[:,2])
plt.plot(target_position[:,1], target_position[:,2])
# v_mea1 = get_v_(r1.data['joint_positions'], r1.data['joint_velocities'], pinrobot.model, pinrobot.model.getFrameId('contact'))
# v_mea2 = get_v_(r2.data['joint_positions'], r2.data['joint_velocities'], pinrobot.model, pinrobot.model.getFrameId('contact'))
# v_mea3 = get_v_(r3.data['joint_positions'], r3.data['joint_velocities'], pinrobot.model, pinrobot.model.getFrameId('contact'))
# v_mea4 = get_v_(r4.data['joint_positions'], r4.data['joint_velocities'], pinrobot.model, pinrobot.model.getFrameId('contact'))
# s.plot_ee_vel( [v_mea1, 
#                 v_mea2,
#                 v_mea3, 
#                 v_mea4],  
#                ['Data 1', 'Data 2', 'Data 3',  'Data 4'], 
#                ['b', 'r', 'g', 'k'], 
#                linestyle=['solid','solid', 'solid', 'solid'])

plt.show()