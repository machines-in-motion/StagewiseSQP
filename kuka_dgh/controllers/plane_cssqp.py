import numpy as np
import pinocchio as pin 

import time

from classical_mpc.ocp import OptimalControlProblemClassicalWithConstraints
from core_mpc import pin_utils
from core_mpc.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger



# @profile
def solveOCP(q, v, ddp, max_sqp_iter, max_qp_iter):
    t = time.time()
    # Update initial state + warm-start
    x = np.concatenate([q, v])
    ddp.problem.x0 = x
    
    xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
    xs_init[0] = x
    us_init = list(ddp.us[1:]) + [ddp.us[-1]] 

    ddp.max_qp_iters = max_qp_iter
    ddp.solve(xs_init, us_init, maxiter=max_sqp_iter, isFeasible=False)
    solve_time = time.time()
    
    return  ddp.us[0], ddp.xs[1], ddp.K[0], solve_time - t, ddp.iter, ddp.cost, ddp.constraint_norm, ddp.gap_norm, ddp.qp_iters, ddp.KKT_norm




class KukaPlaneCSSQP:

    def __init__(self, head, robot, config, run_sim):
        """
        Input:
            head              : thread head
            robot_model       : pinocchio model
            config            : MPC config yaml file
            run_sim           : boolean sim or real
        """
        self.robot   = robot
        self.head    = head
        self.RUN_SIM = run_sim
        self.joint_positions  = head.get_sensor('joint_positions')
        self.joint_velocities = head.get_sensor("joint_velocities")
        self.joint_accelerations = head.get_sensor("joint_accelerations")
        if not self.RUN_SIM:
            self.joint_torques     = head.get_sensor("joint_torques_total")
            self.joint_ext_torques = head.get_sensor("joint_torques_external")
            self.joint_cmd_torques = head.get_sensor("joint_torques_commanded")      


        self.nq = self.robot.model.nq
        self.nv = self.robot.model.nv

        logger.warning("Controlled model dimensions : ")
        logger.warning(" nq = "+str(self.nq))
        logger.warning(" nv = "+str(self.nv))
        
        # Config
        self.config = config
        if(self.RUN_SIM):
            self.q0 = np.asarray(config['q0'])
            self.v0 = self.joint_velocities.copy()  
        else:
            self.q0 = self.joint_positions.copy()
            self.v0 = self.joint_velocities.copy()
        self.x0 = np.concatenate([self.q0, self.v0])
        
        self.Nh = int(self.config['N_h'])
        self.dt_ocp  = self.config['dt']
        self.dt_ctrl = 1./self.config['ctrl_freq']
        self.OCP_TO_CTRL_RATIO = int(self.dt_ocp/self.dt_ctrl)

        # Create OCP 
        self.ddp = OptimalControlProblemClassicalWithConstraints(robot, self.config).initialize(self.x0, callbacks=False)
        self.ddp.regMax = 1e6
        self.ddp.reg_max = 1e6
        self.ddp.termination_tol = self.config['solver_termination_tolerance'] 
        

        # Allocate MPC data
        self.K = self.ddp.K[0]
        self.x_des = self.ddp.xs[0]
        self.tau_ff = self.ddp.us[0]
        self.tau = self.tau_ff.copy() ; self.tau_riccati = np.zeros(self.tau.shape)

        # Initialize torque measurements 
        if(self.RUN_SIM):
            logger.debug("Initial torque measurement signal : simulation --> use u0 = g(q0)")
            self.u0 = pin_utils.get_u_grav(self.q0, self.robot.model, np.zeros(self.robot.model.nq))
            self.joint_torques_total    = self.u0
            self.joint_torques_measured = self.u0
        # DANGER ZONE 
        else:
            logger.warning("Initial torque measurement signal : real robot --> use sensor signal 'joint_torques_total' ")
            self.joint_torques_total    = head.get_sensor("joint_torques_total")
            logger.warning("      >>> Correct minus sign in measured torques ! ")
            self.joint_torques_measured = -self.joint_torques_total 


        self.TASK_PHASE = 0
        self.NH_SIMU    = int(self.Nh*self.dt_ocp/self.dt_ctrl)
        logger.debug("Size of MPC horizon in ctrl cycles = "+str(self.NH_SIMU))
        logger.debug("OCP to ctrl time ratio             = "+str(self.OCP_TO_CTRL_RATIO))

        # Solver logs
        self.t_child         = 0
        self.cost            = 0
        self.cumulative_cost = 0
        self.gap_norm        = np.inf
        self.constraint_norm = np.inf
        self.qp_iters        = 0
        self.KKT             = np.inf


    def warmup(self, thread):
        self.max_sqp_iter = 10  
        self.max_qp_iter  = 100   
        self.u0 = pin_utils.get_u_grav(self.q0, self.robot.model, np.zeros(self.robot.model.nq))
        self.ddp.xs = [self.x0 for i in range(self.config['N_h']+1)]
        self.ddp.us = [self.u0 for i in range(self.config['N_h'])]
        self.tau_ff, self.x_des, self.K, self.t_child, self.ddp_iter, self.cost, self.constraint_norm, self.gap_norm, self.qp_iters, self.KKT = solveOCP(self.joint_positions, 
                                                                                          self.joint_velocities, 
                                                                                          self.ddp, 
                                                                                          self.max_sqp_iter, 
                                                                                          self.max_qp_iter)
        self.cumulative_cost += self.cost
        self.max_sqp_iter = self.config['maxiter']
        self.max_qp_iter  = self.config['max_qp_iter']

    def run(self, thread):        
        # # # # # # # # # 
        # Read sensors  #
        # # # # # # # # # 
        q = self.joint_positions
        v = self.joint_velocities

        # When getting torque measurement from robot, do not forget to flip the sign
        if(not self.RUN_SIM):
            self.joint_torques_measured = -self.joint_torques_total  

        # # # # # # #  
        # Solve OCP #
        # # # # # # #  
        self.tau_ff, self.x_des, self.K, self.t_child, self.ddp_iter, self.cost, self.constraint_norm, self.gap_norm, self.qp_iters, self.KKT = solveOCP(q, 
                                                                                          v, 
                                                                                          self.ddp, 
                                                                                          self.max_sqp_iter, 
                                                                                          self.max_qp_iter)

        # # # # # # # # 
        # Send policy #
        # # # # # # # #
        self.tau = self.tau_ff.copy()

        # Compute gravity
        self.tau_gravity = pin.rnea(self.robot.model, self.robot.data, self.joint_positions, np.zeros(self.nv), np.zeros(self.nv))

        if(self.RUN_SIM == False):
            self.tau -= self.tau_gravity

        ###### DANGER SEND ONLY GRAV COMP
        # self.tau = np.zeros_like(self.tau_full)
        
        self.head.set_control('ctrl_joint_torques', self.tau)     


        pin.framesForwardKinematics(self.robot.model, self.robot.data, q)