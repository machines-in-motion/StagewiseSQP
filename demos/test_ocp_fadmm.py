import pybullet as p
import numpy as np
from dynamic_graph_head import ThreadHead, SimHead, HoldPDController
import pinocchio as pin 

import time

from robot_properties_kuka.config import IiwaConfig
from robot_properties_kuka.iiwaWrapper import IiwaRobot

import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'gnms'
print(python_path)
os.sys.path.insert(1, str(python_path))
from core_mpc import path_utils, sim_utils
from classical_mpc.ocp import OptimalControlProblemClassicalWithConstraints


SIM = False

DGM_PARAMS_PATH = "/home/skleff/ws/workspace/install/robot_properties_kuka/lib/python3.8/site-packages/robot_properties_kuka/robot_properties_kuka/dynamic_graph_manager/dgm_parameters_iiwa.yaml"
CONFIG_NAME = 'kuka_circle_fadmm' 
CONFIG_PATH = 'demos/'+CONFIG_NAME+".yml"


pin_robot = IiwaConfig.buildRobotWrapper()
config = path_utils.load_yaml_file(CONFIG_PATH)

# Make OCP 
q_init = np.asarray(config['q0'] )
v_init = np.asarray(config['dq0'])
x0 = np.concatenate([q_init, v_init])
ddp = OptimalControlProblemClassicalWithConstraints(pin_robot, config).initialize(x0, callbacks=True)

# Solve
qp_iters = 1000
sqp_ites = 100
ddp.with_callbacks = True
ddp.use_filter_ls = True
ddp.filter_size = 10
ddp.termination_tol = 1e-3
ddp.warm_start = True
ddp.max_qp_iters = qp_iters

ddp.xs = [x0]*(config['N_h']+1)
ddp.solve()