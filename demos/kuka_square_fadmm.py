import pybullet as p
import numpy as np
from dynamic_graph_head import ThreadHead, SimHead, HoldPDController
import pinocchio as pin 

import time

import dynamic_graph_manager_cpp_bindings
from robot_properties_kuka.config import IiwaConfig
from robot_properties_kuka.iiwaWrapper import IiwaRobot
from bullet_utils.env import BulletEnvWithGround

import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'gnms'
print(python_path)
os.sys.path.insert(1, str(python_path))
from controllers.kuka_square_fadmm import KukaSquareFADMM
from core_mpc import path_utils, sim_utils


SIM = True

DGM_PARAMS_PATH = "/home/ajordana/ws/workspace/install/robot_properties_kuka/lib/python3.8/site-packages/robot_properties_kuka/robot_properties_kuka/dynamic_graph_manager/dgm_parameters_iiwa.yaml"
CONFIG_NAME = 'kuka_square_fadmm' 
CONFIG_PATH = 'demos/'+CONFIG_NAME+".yml"


pin_robot = IiwaConfig.buildRobotWrapper()
config = path_utils.load_yaml_file(CONFIG_PATH)


# SIMULATION
if SIM:
    env = BulletEnvWithGround(p.GUI)
    robot_simulator = env.add_robot(IiwaRobot())
    robot_simulator.pin_robot = pin_robot
    q_init = np.asarray(config['q0'] )
    v_init = np.asarray(config['dq0'])
    v_init = np.zeros(pin_robot.model.nv) 
    robot_simulator.reset_state(q_init, v_init)
    # Display the target 
    p_ball = np.asarray(config['frameTranslationRef'])
    sim_utils.display_ball(p_ball, robot_base_pose= pin.SE3.Identity(), RADIUS=.05, COLOR=[1.,0.,0.,.6])
    head = SimHead(robot_simulator, with_sliders=False)
# !! REAL ROBOT !!
else:
    path = DGM_PARAMS_PATH 
    head = dynamic_graph_manager_cpp_bindings.DGMHead(path)
    target = None
    env = None



ctrl = KukaSquareFADMM(head, pin_robot, config, run_sim=SIM)
# ctrl.warm_start(100)
# ctrl.update_desired_position(x_des)




thread_head = ThreadHead(
    1./config['simu_freq'],                                         # dt.
    HoldPDController(head, 50., 0.5, with_sliders=False),           # Safety controllers.
    head,                                                           # Heads to read / write from.
    [], # [('vicon', Vicon('172.24.117.119:801', ['cube10/cube10']))], 
    env                                                             # Environment to step.
)



thread_head.switch_controllers(ctrl)

if(config['USE_PROXQP']):
    suffix = 'PROXQP'
else:
    suffix = 'FADMM'



thread_head.switch_controllers(ctrl)

if(config['USE_PROXQP']):
    suffix = 'PROXQP'
else:
    suffix = 'FADMM'

if SIM:
    thread_head.start_logging(10, "/home/ajordana/Desktop/FADMM_demos/square/SIM_"+str(time.time())+".mds")
    thread_head.sim_run_timed(100000)
    # thread_head.stop_logging()
    thread_head.plot_timing()
else:
    thread_head.start()
    thread_head.start_logging(30, "/home/ajordana/Desktop/FADMM_demos/square/REAL_"+str(time.time())+".mds")
    time.sleep(30)
    # thread_head.plot_timing()
# ctrl.bench.plot_timer()