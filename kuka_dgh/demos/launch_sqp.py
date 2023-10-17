import pybullet as p
import numpy as np
from dynamic_graph_head import ThreadHead, SimHead, HoldPDController
import pinocchio as pin 

from datetime import datetime

import dynamic_graph_manager_cpp_bindings
from robot_properties_kuka.config import IiwaConfig
from robot_properties_kuka.iiwaWrapper import IiwaRobot
from bullet_utils.env import BulletEnvWithGround

import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'StagewiseSQP'
os.sys.path.insert(1, str(python_path))

from core_mpc import path_utils, sim_utils


SUPPORTED_EXPERIMENTS = ['reach_ssqp', 'circle_ssqp', 'circle_cssqp', 'square_cssqp']

SIM = True

DGM_PARAMS_PATH = "/home/skleff/ws/workspace/install/robot_properties_kuka/lib/python3.8/site-packages/robot_properties_kuka/robot_properties_kuka/dynamic_graph_manager/dgm_parameters_iiwa.yaml"
CONFIG_NAME = 'circle_ssqp' # circle_sqp
CONFIG_PATH = 'config/'+CONFIG_NAME+".yml"

try: 
    assert(CONFIG_NAME in SUPPORTED_EXPERIMENTS)
except NameError:
    print("Error : config file name must be in "+str(SUPPORTED_EXPERIMENTS))
    

if(CONFIG_NAME == 'reach_ssqp'):
    from controllers.reach_ssqp import KukaReachSSQP as MPCController
elif(CONFIG_NAME == 'circle_ssqp'):
    from controllers.circle_ssqp import KukaCircleSSQP as MPCController
elif(CONFIG_NAME == 'circle_cssqp'):
    from controllers.circle_cssqp import KukaCircleCSSQP as MPCController
elif(CONFIG_NAME == 'square_cssqp'):
    from controllers.square_cssqp import KukaSquareCSSQP as MPCController


pin_robot = IiwaConfig.buildRobotWrapper()
config = path_utils.load_yaml_file(CONFIG_PATH)



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
    # Display the target 
    p_ball = np.asarray(config['frameTranslationRef'])
    sim_utils.display_ball(p_ball, robot_base_pose= pin.SE3.Identity(), RADIUS=.05, COLOR=[1.,0.,0.,.6])
    head = SimHead(robot_simulator, with_sliders=False)
# !! REAL ROBOT !!
else:
    config['T_tot'] = 400              
    path = DGM_PARAMS_PATH 
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


prefix = "/tmp/"
suffix = "_"+config['SOLVER']

# LOG_FIELDS = ['KKT',
#               'ddp_iter',
#               't_child',
#               'joint_positions',
#               'target_position']

if SIM:
    thread_head.start_logging(int(config['T_tot']), prefix+CONFIG_NAME+"_SIM_"+str(datetime.now().isoformat())+suffix+".mds") #, LOG_FIELDS=LOG_FIELDS)
    thread_head.sim_run_timed(int(config['T_tot']))
    thread_head.stop_logging()
else:
    thread_head.start()
    thread_head.start_logging(15, prefix+CONFIG_NAME+"_REAL_"+str(datetime.now().isoformat())+suffix+".mds", LOG_FIELDS=LOG_FIELDS)
    
thread_head.plot_timing()
