import pybullet as p
import numpy as np
from dynamic_graph_head import ThreadHead, SimHead, HoldPDController
from datetime import datetime
import dynamic_graph_manager_cpp_bindings
from mim_robots.robot_loader import load_bullet_wrapper, load_pinocchio_wrapper
from mim_robots.pybullet.env import BulletEnvWithGround
from mim_robots.robot_list import MiM_Robots
import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'StagewiseSQP'
os.sys.path.insert(1, str(python_path))
import pinocchio as pin

class KukaZeroTorque:

    def __init__(self, head, pin_robot, run_sim):
        """
        Input:
            head              : thread head
            pin_robot         : pinocchio wrapper
            run_sim           : boolean sim or real
        """
        self.robot   = pin_robot
        self.head    = head
        self.RUN_SIM = run_sim
        self.joint_positions  = head.get_sensor('joint_positions')
        self.joint_velocities = head.get_sensor("joint_velocities")
        self.joint_accelerations = head.get_sensor("joint_accelerations")
        if not self.RUN_SIM:
            self.joint_torques_total = head.get_sensor("joint_torques_total")
            self.joint_ext_torques   = head.get_sensor("joint_torques_external")
            self.joint_cmd_torques   = head.get_sensor("joint_torques_commanded")      
        self.nq = self.robot.model.nq
        self.nv = self.robot.model.nv
        self.tau_gravity = np.zeros(self.nq)
        self.tau = np.zeros_like(self.tau_gravity)
        
    def warmup(self, thread):
        pass

    def run(self, thread):        

        # # # # # # # # # 
        # Read sensors  #
        # # # # # # # # # 
        q = self.joint_positions
        v = self.joint_velocities
        # Torque measurements on robot have flipped axis -> flip sign
        if(self.RUN_SIM == False):
            self.joint_torques_measured = -self.joint_torques_total  
            
        # # # # # # # # 
        # Send policy #
        # # # # # # # #
        # Compute gravity
        self.tau_gravity = pin.rnea(self.robot.model, self.robot.data, self.joint_positions, np.zeros(self.nv), np.zeros(self.nv))
        # Zero torque
        self.tau = np.zeros_like(self.tau_gravity)
        # Add gravity ONLY in simulation (real KUKA robot already compensates gravity)
        if(self.RUN_SIM == True):
            self.tau += self.tau_gravity
        self.head.set_control('ctrl_joint_torques', self.tau)    
        # Update kinematics 
        pin.framesForwardKinematics(self.robot.model, self.robot.data, q)


# # # # # # # # # # # # #
# EXPERIMENT PARAMETERS #
# # # # # # # # # # # # #
SIM       = False
CTRL_FREQ = 1000.
T_TOT     = 15.



# # # # # # # # # # #
# LOAD ROBOT MODEL  #
# # # # # # # # # # #
pin_robot = load_pinocchio_wrapper('iiwa')


# # # # # # # # # # # # #
# Setup control thread  #
# # # # # # # # # # # # #
if SIM:
    # Sim env + set initial state 
    env = BulletEnvWithGround(p.GUI)
    robot_simulator = load_bullet_wrapper('iiwa')
    env.add_robot(robot_simulator)
    q_init = np.asarray( [0., 1.0471975511965976, 0., -1.1344640137963142, 0.2,  0.7853981633974483, 0.]   )
    v_init = np.zeros_like(q_init) 
    robot_simulator.reset_state(q_init, v_init)
    robot_simulator.forward_robot(q_init, v_init)
    # <<<<< Customize your PyBullet environment here if necessary
    head = SimHead(robot_simulator, with_sliders=False)
# !!!!!!!!!!!!!!!!
# !! REAL ROBOT !!
# !!!!!!!!!!!!!!!!
else:
    path = MiM_Robots['iiwa'].dgm_path  
    print(path)
    head = dynamic_graph_manager_cpp_bindings.DGMHead(path)
    target = None
    env = None

ctrl = KukaZeroTorque(head, pin_robot, run_sim=SIM)

thread_head = ThreadHead(
    1./CTRL_FREQ,                                         # dt.
    HoldPDController(head, 50., 0.5, with_sliders=False),           # Safety controllers.
    head,                                                           # Heads to read / write from.
    [], 
    env                                                             # Environment to step.
)

thread_head.switch_controllers(ctrl)





# # # # # # # # #
# Data logging  #
# # # # # # # # # 
prefix     = "/tmp/"
suffix     = "_test"
LOG_FIELDS = ['joint_positions',
              'joint_velocities',
              'tau',
              'tau_gravity',
              'joint_torques_measured',
              'joint_cmd_torques']


# # # # # # # # # # # 
# Launch experiment #
# # # # # # # # # # # 
if SIM:
    thread_head.start_logging(int(T_TOT), prefix+"_SIM_"+str(datetime.now().isoformat())+suffix+".mds", LOG_FIELDS=LOG_FIELDS)
    thread_head.sim_run_timed(int(T_TOT))
    thread_head.stop_logging()
else:
    thread_head.start()
    thread_head.start_logging(T_TOT, prefix+"_REAL_"+str(datetime.now().isoformat())+suffix+".mds", LOG_FIELDS=LOG_FIELDS)
    
thread_head.plot_timing() # <<<<<<<<<<<<< Comment out to skip timings plot
