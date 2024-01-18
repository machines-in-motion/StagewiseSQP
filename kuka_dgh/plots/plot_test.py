from plot_utils import SimpleDataPlotter
from mim_data_utils import DataReader
from croco_mpc_utils.pinocchio_utils import *
import numpy as np
import matplotlib.pyplot as plt 

from mim_robots.robot_loader import load_pinocchio_wrapper


# # # # # # # # # # # # #
# EXPERIMENT PARAMETERS #
# # # # # # # # # # # # #
SIM       = False
CTRL_FREQ = 1000.
T_TOT     = 15.


# # # # # # # # # # #
# LOAD ROBOT MODEL  #
# # # # # # # # # # #
pin_robot    = load_pinocchio_wrapper('iiwa')
model        = pin_robot.model
data         = model.createData()
frameId      = model.getFrameId('contact')
nq = model.nq ; nv = model.nv
# Overwrite effort limit as in DGM
model.effortLimit = np.array([100, 100, 50, 50, 20, 10, 10])



# # # # # # # # # # # # # # # #
# DATA PLOTTER AND SAVE PATH  #
# # # # # # # # # # # # # # # #
s = SimpleDataPlotter(dt=1./CTRL_FREQ)
if(SIM):
    data_path = '/tmp/'
    data_name = '_SIM_2024-01-18T12:01:02.780621_test' 
else:
    data_path = '/tmp/'
    data_name = '_REAL_2024-01-18T12:28:19.351083_test' 
r = DataReader(data_path+data_name+'.mds')
N = r.data['absolute_time'].shape[0]
print("Total number of control cycles = ", N)
    
# # # # # # # # # #
# JOINT POSITIONS #
# # # # # # # # # #
s.plot_joint_pos( [r.data['joint_positions']], 
                  ['Measured'], 
                  ['r'],
                  ylims=[model.lowerPositionLimit, model.upperPositionLimit],
                  linestyle=['solid'] )

# # # # # # # # #
# JOINT TORQUES #
# # # # # # # # #
if(SIM):
    s.plot_joint_tau( [r.data['tau']],
                      ['total'], 
                      ['r'],
                      ylims=[-model.effortLimit, +model.effortLimit] )
else:
    # Our self.tau was subtracted gravity, so we add it again
    # joint_torques_measured DOES include the gravity torque from KUKA
    # There is a sign mismatch in the axis so we use a minus sign
    s.plot_joint_tau( [-r.data['joint_cmd_torques'], 
                       r.data['joint_torques_measured'], 
                       r.data['tau'] + r.data['tau_gravity']], 
                      ['-cmd (FRI)', 
                       'Measured', 
                       'Desired (sent to robot) [+g(q)]'], 
                      ['k', 'r', 'b'],
                      ylims=[-model.effortLimit, +model.effortLimit],
                      linestyle=['dotted', 'solid', 'solid'] )


# # # # # # # # # # #
# END-EFF POSITION  #
# # # # # # # # # # #
p_mea = get_p_(r.data['joint_positions'], model, model.getFrameId('contact'))
s.plot_ee_pos([p_mea],  
              ['Measured'], 
              ['r'], 
              linestyle=['solid'])
plt.show()