import numpy as np
import pinocchio as pin 
import mim_solvers

import time

from croco_mpc_utils.ocp_constraints import OptimalControlProblemClassicalWithConstraints
import croco_mpc_utils.pinocchio_utils as pin_utils

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger



# @profile
def solveOCP(q, v, solver, max_sqp_iter, max_qp_iter, target_reach, TASK_PHASE):
    t = time.time()
    # Update initial state + warm-start
    x = np.concatenate([q, v])
    solver.problem.x0 = x
    
    xs_init = list(solver.xs[1:]) + [solver.xs[-1]]
    xs_init[0] = x
    us_init = list(solver.us[1:]) + [solver.us[-1]] 
    
    # Update OCP 
    if(TASK_PHASE == 1):
        # Updates nodes between node_id and terminal node 
        for k in range( solver.problem.T ):
            solver.problem.runningModels[k].differential.costs.costs["translation"].active = True
            solver.problem.runningModels[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
            solver.problem.runningModels[k].differential.costs.costs["translation"].weight = 25.
        solver.problem.terminalModel.differential.costs.costs["translation"].active = True
        solver.problem.terminalModel.differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
        solver.problem.terminalModel.differential.costs.costs["translation"].weight = 25.

    solver.max_qp_iters = max_qp_iter
    solver.solve(xs_init, us_init, maxiter=max_sqp_iter, isFeasible=False)
    solve_time = time.time()
    
    return  solver.us[0], solver.xs[1], solver.K[0], solve_time - t, solver.iter, solver.cost, solver.constraint_norm, solver.gap_norm, solver.qp_iters, solver.KKT_norm




class KukaCircleCSSQP:

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
        problem = OptimalControlProblemClassicalWithConstraints(robot, self.config).initialize(self.x0)
        # Initialize the solver
        if(config['SOLVER'] == 'proxqp'):
            logger.warning("Using the ProxQP solver.")
            self.solver = mim_solvers.SolverProxQP(problem)
        elif(config['SOLVER'] == 'cssqp'):
            logger.warning("Using the CSSQP solver.")
            self.solver = mim_solvers.SolverCSQP(problem)
        self.solver.with_callbacks  = self.config['with_callbacks']
        self.solver.use_filter_ls   = self.config['use_filter_ls']
        self.solver.filter_size     = self.config['filter_size']
        self.solver.warm_start      = self.config['warm_start']
        self.solver.termination_tol = self.config['solver_termination_tolerance']
        self.solver.max_qp_iters    = self.config['max_qp_iter']
        self.solver.eps_abs         = self.config['qp_termination_tol_abs']
        self.solver.eps_rel         = self.config['qp_termination_tol_rel']
        self.solver.warm_start_y    = self.config['warm_start_y']
        self.solver.reset_rho       = self.config['reset_rho']  
        self.solver.regMax = 1e6
        self.solver.reg_max = 1e6
        

        # Allocate MPC data
        self.K = self.solver.K[0]
        self.x_des = self.solver.xs[0]
        self.tau_ff = self.solver.us[0]
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


        # Circle trajectory 
        N_total_pos = int((self.config['T_tot'] - self.config['T_REACH'])/self.dt_ctrl + self.Nh*self.OCP_TO_CTRL_RATIO)
        N_circle    = int((self.config['T_tot'] - self.config['T_CIRCLE'])/self.dt_ctrl + self.Nh*self.OCP_TO_CTRL_RATIO )
        self.target_position_traj = np.zeros( (N_total_pos, 3) )
        # absolute desired position
        self.pdes = np.asarray(self.config['frameTranslationRef']) 
    
        radius = 0.3 ; omega = 2.
        self.target_position_traj[0:N_circle, :] = [np.array([self.pdes[0],
                                                              self.pdes[1] + radius * np.sin(i*self.dt_ctrl*omega), 
                                                              self.pdes[2] + radius * (1-np.cos(i*self.dt_ctrl*omega)) ]) for i in range(N_circle)]
        self.target_position_traj[N_circle:, :] = self.target_position_traj[N_circle-1,:]
        # Targets over one horizon (initially = absolute target position)
        self.target_position = np.zeros((self.Nh+1, 3)) 
        self.target_position[:,:] = self.pdes.copy() 
        self.target_position_x = self.target_position[:,0] 
        self.target_position_y = self.target_position[:,1] 
        self.target_position_z = self.target_position[:,2]

        self.TASK_PHASE      = 0
        self.NH_SIMU   = int(self.Nh*self.dt_ocp/self.dt_ctrl)
        self.T_CIRCLE  = int(self.config['T_CIRCLE']/self.dt_ctrl)
        logger.debug("Size of MPC horizon in ctrl cycles = "+str(self.NH_SIMU))
        logger.debug("Start of circle phase in ctrl cycles = "+str(self.T_CIRCLE))
        logger.debug("OCP to ctrl time ratio = "+str(self.OCP_TO_CTRL_RATIO))

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
        self.solver.xs = [self.x0 for i in range(self.config['N_h']+1)]
        self.solver.us = [self.u0 for i in range(self.config['N_h'])]
        self.tau_ff, self.x_des, self.K, self.t_child, self.ddp_iter, self.cost, self.constraint_norm, self.gap_norm, self.qp_iters, self.KKT = solveOCP(self.joint_positions, 
                                                                                          self.joint_velocities, 
                                                                                          self.solver, 
                                                                                          self.max_sqp_iter, 
                                                                                          self.max_qp_iter, 
                                                                                          self.target_position,
                                                                                          self.TASK_PHASE)
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

        # # # # # # # # # 
        # # Update OCP  #
        # # # # # # # # # 
        time_to_circle  = int(thread.ti - self.T_CIRCLE)   


        if(time_to_circle == 0): 
            print("Entering circle phase")
        # If circle tracking phase enters the MPC horizon, start updating models from the end with tracking models      
        if(0 <= time_to_circle and time_to_circle <= self.NH_SIMU):
            self.TASK_PHASE = 1


        if(0 <= time_to_circle):
            # set position refs over current horizon
            tf  = time_to_circle + (self.Nh+1)*self.OCP_TO_CTRL_RATIO
            # Target in (x,y)  = circle trajectory 
            self.target_position = self.target_position_traj[time_to_circle:tf:self.OCP_TO_CTRL_RATIO,:]
            # Record target signals
            self.target_position_x = self.target_position[:,0] 
            self.target_position_y = self.target_position[:,1] 
            self.target_position_z = self.target_position[:,2]

        # # # # # # #  
        # Solve OCP #
        # # # # # # #  
        self.tau_ff, self.x_des, self.K, self.t_child, self.ddp_iter, self.cost, self.constraint_norm, self.gap_norm, self.qp_iters, self.KKT = solveOCP(q, 
                                                                                          v, 
                                                                                          self.solver, 
                                                                                          self.max_sqp_iter, 
                                                                                          self.max_qp_iter, 
                                                                                          self.target_position,
                                                                                          self.TASK_PHASE)

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