import pybullet as p
import numpy as np
from dynamic_graph_head import ThreadHead, SimHead, HoldPDController
import pinocchio as pin 

import time

import dynamic_graph_manager_cpp_bindings
from robot_properties_kuka.config import IiwaConfig, IiwaReducedConfig
from robot_properties_kuka.iiwaWrapper import IiwaRobot
from robot_properties_kuka.iiwaReducedWrapper import IiwaReducedRobot
from bullet_utils.env import BulletEnvWithGround

import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'gnms'
print(python_path)
os.sys.path.insert(1, str(python_path))
from controllers.kuka_circle_gnms import KukaCircleGNMS
from core_mpc import path_utils, sim_utils

SIM = True

DGM_PARAMS_PATH = "/home/skleff/ws/workspace/install/robot_properties_kuka/lib/python3.8/site-packages/robot_properties_kuka/robot_properties_kuka/dynamic_graph_manager/dgm_parameters_iiwa.yaml"
# DGM_PARAMS_PATH = "/ws/workspace/install/robot_properties_kuka/lib/python3.8/site-packages/robot_properties_kuka/robot_properties_kuka/dynamic_graph_manager/dgm_parameters_iiwa_reduced.yaml"
CONFIG_NAME = 'kuka_circle_gnms' # reaching_mpc_reduced
CONFIG_PATH = 'demos/'+CONFIG_NAME+".yml"


# controlled_joints = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
# qref =  np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.])
# pin_robot = IiwaReducedConfig.buildRobotWrapper(controlled_joints, qref)
pin_robot = IiwaConfig.buildRobotWrapper()
config = path_utils.load_yaml_file(CONFIG_PATH)



# SIMULATION
if SIM:
    env = BulletEnvWithGround(p.GUI)
    robot_simulator = env.add_robot(IiwaRobot())
    robot_simulator.pin_robot = pin_robot
    # robot = IiwaReducedRobot(controlled_joints, qref)
    q_init = np.zeros(pin_robot.model.nq) 
    q_init[1] =0.34
    q_init[3] = -0.75
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



ctrl = KukaCircleGNMS(head, pin_robot, config, run_sim=SIM, use_gnms=False)
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

if SIM:
    # thread_head.start_logging(6, "test.mds")
    thread_head.start_logging(10, "/tmp/kuka_circle_gnms_sim_FDDP.mds")
    thread_head.sim_run_timed(100000)
    # thread_head.stop_logging()
    thread_head.plot_timing()
else:
    thread_head.start()
    thread_head.start_logging(15, "/tmp/kuka_circle_gnms_real.mds")
    # time.sleep(30)
    # thread_head.plot_timing()
# ctrl.bench.plot_timer()