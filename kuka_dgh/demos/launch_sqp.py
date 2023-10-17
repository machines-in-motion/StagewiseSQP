import pybullet as p
import numpy as np
from dynamic_graph_head import ThreadHead, SimHead, HoldPDController
from datetime import datetime
import dynamic_graph_manager_cpp_bindings
from robot_properties_kuka.config import IiwaConfig
from robot_properties_kuka.iiwaWrapper import IiwaRobot
from bullet_utils.env import BulletEnvWithGround
import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'StagewiseSQP'
os.sys.path.insert(1, str(python_path))

import launch_utils





# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Choose experiment, load config and import controller  #  
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
SIM                     = True
EXP_NAME                = 'plane_cssqp' # <<<<<<<<<<<<< Choose experiment here (cf. launch_utils)
config, MPCController   = launch_utils.load_config_file_and_import_controller(EXP_NAME)
    
    
    
    
    
# # # # # # # # # # # #
# Import robot model  #
# # # # # # # # # # # #
pin_robot = IiwaConfig.buildRobotWrapper()





# # # # # # # # # # # # #
# Setup control thread  #
# # # # # # # # # # # # #
if SIM:
    # Sim env + set initial state 
    config['T_tot'] = 15              
    env = BulletEnvWithGround(p.GUI)
    robot_simulator = env.add_robot(IiwaRobot())
    robot_simulator.pin_robot = pin_robot
    q_init = np.asarray(config['q0'] )
    v_init = np.asarray(config['dq0'])
    robot_simulator.reset_state(q_init, v_init)
    robot_simulator.forward_robot(q_init, v_init)
    # <<<<< Customize your PyBullet environment here if necessary
    head = SimHead(robot_simulator, with_sliders=False)
# !!!!!!!!!!!!!!!!
# !! REAL ROBOT !!
# !!!!!!!!!!!!!!!!
else:
    config['T_tot'] = 400              
    path = IiwaConfig.paths['dgm_yaml']  
    head = dynamic_graph_manager_cpp_bindings.DGMHead(path)
    target = None
    env = None

ctrl = MPCController(head, pin_robot, config, run_sim=SIM)

thread_head = ThreadHead(
    1./config['ctrl_freq'],                                         # dt.
    HoldPDController(head, 50., 0.5, with_sliders=False),           # Safety controllers.
    head,                                                           # Heads to read / write from.
    [], 
    env                                                             # Environment to step.
)

thread_head.switch_controllers(ctrl)





# # # # # # # # #
# Data logging  #
# # # # # # # # # <<<<<<<<<<<<< Choose data save path & log config here (cf. launch_utils)
prefix     = "/tmp/"
suffix     = "_"+config['SOLVER']
LOG_FIELDS = launch_utils.get_log_config(EXP_NAME) 
# LOG_FIELDS = launch_utils.LOGS_NONE 
# LOG_FIELDS = launch_utils.SSQP_LOGS_MINIMAL 
# LOG_FIELDS = launch_utils.CSSQP_LOGS_MINIMAL 







# # # # # # # # # # # 
# Launch experiment #
# # # # # # # # # # # 
if SIM:
    thread_head.start_logging(int(config['T_tot']), prefix+EXP_NAME+"_SIM_"+str(datetime.now().isoformat())+suffix+".mds", LOG_FIELDS=LOG_FIELDS)
    thread_head.sim_run_timed(int(config['T_tot']))
    thread_head.stop_logging()
else:
    thread_head.start()
    thread_head.start_logging(15, prefix+EXP_NAME+"_REAL_"+str(datetime.now().isoformat())+suffix+".mds", LOG_FIELDS=LOG_FIELDS)
    
thread_head.plot_timing() # <<<<<<<<<<<<< Comment out to skip timings plot
