import numpy as np
import pinocchio as pin 

import time

from classical_mpc.ocp import OptimalControlProblemClassicalWithConstraints
from core_mpc import pin_utils
from core_mpc.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


USE_PIPE = False


def solveOCP(q, v, ddp, max_sqp_iter, max_qp_iter, node_id_circle, target_reach, TASK_PHASE):
    # Read state last measurement from parent process
    t = time.time()
    x = np.concatenate([q, v])
    # Update initial state + warm-start
    ddp.problem.x0 = x
    xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
    xs_init[0] = x
    us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
    # Warm start the Lagrange multipliers and rho
    # logger.warning("warm_start_y ="+str(ddp.warm_start_y))
    # logger.warning("reset_rho ="+str(ddp.reset_rho))
    # Update OCP for reaching phase
    m = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
    if(TASK_PHASE == 1):
        # If node id is valid
        if(node_id_circle <= ddp.problem.T and node_id_circle >= 0):
            # Updates nodes between node_id and terminal node 
            for k in range( node_id_circle, ddp.problem.T+1, 1 ):
                m[k].differential.costs.costs["translation"].active = True
                m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                m[k].differential.costs.costs["translation"].weight = 25.
    problem_formulation_time = time.time()
    t_child_1 =  problem_formulation_time - t
    # Solve OCP 
    ddp.max_qp_iters = max_qp_iter
    ddp.solve(xs_init, us_init, maxiter=max_sqp_iter, isFeasible=False)
    solve_time = time.time()
    ddp_iter = ddp.iter
    t_child =  solve_time - problem_formulation_time
    cost = ddp.cost
    constraint_norm = ddp.constraint_norm
    gap_norm = ddp.gap_norm
    qp_iters = ddp.qp_iters
    kkt_norm = ddp.KKT_norm
    return ddp.us, ddp.xs, ddp.K, t_child, ddp_iter, t_child_1, cost, constraint_norm, gap_norm, qp_iters, kkt_norm


def rt_SolveOCP(child_conn, ddp, max_sqp_iter, max_qp_iter, node_id_circle, target_reach, TASK_PHASE):
    while True:
        # Read state last measurement from parent process
        q, v, node_id_circle, target_reach, TASK_PHASE = child_conn.recv()
        t = time.time()
        # Update initial state + warm-start
        x = np.concatenate([q, v])
        ddp.problem.x0 = x
        xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
        xs_init[0] = x
        us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
        # Get OCP nodes
        m = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
        # Update OCP for reaching phase
        if(TASK_PHASE == 1):
            # If node id is valid
            if(node_id_circle <= ddp.problem.T and node_id_circle >= 0):
                # Updates nodes between node_id and terminal node 
                for k in range( node_id_circle, ddp.problem.T+1, 1 ):
                    m[k].differential.costs.costs["translation"].active = True
                    m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                    # m[k].differential.costs.costs["translation"].weight = w
        problem_formulation_time = time.time()
        t_child_1 =  problem_formulation_time - t
        # Solve OCP 
        ddp.max_qp_iters = max_qp_iter
        ddp.solve(xs_init, us_init, maxiter=max_sqp_iter, isFeasible=False)
        # Send solution to parent process + riccati gains
        solve_time = time.time()
        ddp_iter = ddp.iter
        t_child =  solve_time - problem_formulation_time
        cost = ddp.cost
        constraint_norm = ddp.constraint_norm
        gap_norm = ddp.gap_norm
        qp_iters = ddp.qp_iters
        kkt_norm = ddp.KKT_norm
        child_conn.send((ddp.us, ddp.xs, ddp.K, t_child, ddp_iter, t_child_1, cost, constraint_norm, gap_norm, qp_iters, kkt_norm))        



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
        if not self.RUN_SIM:
            self.joint_torques = head.get_sensor("joint_torques_total")
            self.joint_ext_torques = head.get_sensor("joint_torques_external")
            self.joint_cmd_torques = head.get_sensor("joint_torques_commanded")      

        if(USE_PIPE):
            self.parent_conn, self.child_conn = Pipe()
            self.sent = False

        self.nq = self.robot.model.nq
        self.nv = self.robot.model.nv

        # Get config + initial state (from sim or sensors)
        self.config = config
        if(self.RUN_SIM):
            self.q0 = np.asarray(config['q0'])
        else:
            self.q0 = self.joint_positions   
        self.endeffFrameId = self.robot.model.getFrameId('contact') 
        self.v0 = self.joint_velocities  
        self.x0 = np.concatenate([self.q0, self.v0]) 
        self.Nh = self.config['N_h']
        self.dt_ocp  = self.config['dt']
        self.dt_plan = 1./self.config['plan_freq']
        self.dt_simu = 1./self.config['simu_freq']
        self.ocp_to_sim_ratio = 1. / ( self.config['simu_freq'] * self.dt_ocp )
        self.sim_to_plan_ratio = self.config['simu_freq']/self.config['plan_freq']
        # Create OCP
        self.ddp = OptimalControlProblemClassicalWithConstraints(robot, self.config).initialize(self.x0, callbacks=False)
        self.ug  = pin_utils.get_u_grav(self.x0[:self.robot.model.nq], self.robot.model, self.config['armature'])


        # Allocate MPC data
        self.us = self.ddp.us ; self.xs = self.ddp.xs ; self.Ks = self.ddp.K 
        self.x = self.xs[0] ; self.tau_ff = self.us[0] ; self.K = self.Ks[0]
        self.tau = self.tau_ff.copy() ; self.tau_riccati = np.zeros(self.tau.shape)
        self.x1 = self.xs[1]
        self.nb_ctrl = 0
        self.nb_plan = 0

        # Initialize torque measurements 
        if(self.RUN_SIM):
            logger.debug("Initial torque measurement signal : simulation --> use u0 = g(q0)")
            self.u0 = self.ug
            self.joint_torques_total    = self.u0
            self.joint_torques_measured = self.u0
        # DANGER ZONE 
        else:
            logger.warning("Initial torque measurement signal : real robot --> use sensor signal 'joint_torques_total' ")
            self.joint_torques_total    = head.get_sensor("joint_torques_total")
            logger.warning("      >>> Correct minus sign in measured torques ! ")
            self.joint_torques_measured = -self.joint_torques_total 


        # self.target_position = np.asarray(self.config['contactPosition']) + np.asarray(self.config['oPc_offset'])
        # Circle trajectory 
        N_total_pos = int((self.config['T_tot'] - 0.)/self.dt_ocp + self.Nh)
        N_circle = int((self.config['T_tot'] - self.config['T_CIRCLE'])/self.dt_ocp) + self.Nh
        self.target_position_traj = np.zeros( (N_total_pos, 3) )
        # absolute desired position
        self.pdes = np.asarray(self.config['frameTranslationRef']) 
        radius = 0.3 ; omega = 2.
        self.target_position_traj[0:N_circle, :] = [np.array([self.pdes[0],
                                                              self.pdes[1] + radius * np.sin(i*self.dt_ocp*omega), 
                                                              self.pdes[2] + radius * (1-np.cos(i*self.dt_ocp*omega)) ]) for i in range(N_circle)]
        self.target_position_traj[N_circle:, :] = self.target_position_traj[N_circle-1,:]
        # plt.plot(self.target_position_traj, label='pos')
        # plt.show()
        # Targets over one horizon (initially = absolute target position)
        self.target_position = np.zeros((self.Nh+1, 3)) 
        self.target_position[:,:] = self.pdes.copy() 
        self.target_position_x = self.target_position[:,0] 
        self.target_position_y = self.target_position[:,1] 
        self.target_position_z = self.target_position[:,2]

        self.node_id_circle = -1
        self.TASK_PHASE = 0
        self.NH_SIMU   = int(self.Nh*self.dt_ocp/self.dt_simu)
        self.T_CIRCLE = int(self.config['T_CIRCLE']/self.dt_simu)
        self.OCP_TO_SIMU_CYCLES = 1./(self.dt_simu / self.dt_ocp)
        logger.debug("Size of MPC horizon in simu cycles = "+str(self.NH_SIMU))
        logger.debug("Start of circle phase in simu cycles = "+str(self.T_CIRCLE))
        logger.debug("OCP to SIMU time ratio = "+str(self.OCP_TO_SIMU_CYCLES))
        self.cumulative_cost = 0

        # Solver logs
        self.gap_norm = np.inf
        self.constraint_norm = np.inf
        self.qp_iters = 0
        self.kkt_norm = np.inf


    def warmup(self, thread):
        self.max_sqp_iter = 10  
        self.max_qp_iters  = 100   
        self.ddp.xs = [self.x0 for i in range(self.config['N_h']+1)]
        self.ddp.us = [self.ug for i in range(self.config['N_h'])]
        self.is_plan_updated = False
        # USE pipe
        if(USE_PIPE):
            self.subp = Process(target=rt_SolveOCP, args=(self.child_conn, self.ddp, self.max_sqp_iter, self.max_qp_iters, self.node_id_circle, self.target_position, self.TASK_PHASE))
            self.subp.start()
            # Read sensors and publish real state 
            self.parent_conn.send((self.joint_positions, self.joint_velocities, self.target_position))
            self.us, self.xs, self.Ks, self.t_child, self.ddp_iter, self.t_child_1, self.cost, self.constraint_norm, self.gap_norm, self.qp_iters, self.kkt_norm  = self.parent_conn.recv()
        # NO pipe
        else:
            self.us, self.xs, self.Ks, self.t_child, self.ddp_iter, self.t_child_1, self.cost, self.constraint_norm, self.gap_norm, self.qp_iters, self.kkt_norm = solveOCP(self.joint_positions, 
                                                                                            self.joint_velocities, 
                                                                                            self.ddp, 
                                                                                            self.max_sqp_iter, 
                                                                                            self.max_qp_iters,
                                                                                            self.node_id_circle,
                                                                                            self.target_position,
                                                                                            self.TASK_PHASE)
        self.cumulative_cost += self.cost
        self.check = 0
        self.max_sqp_iter = self.config['maxiter']
        self.max_qp_iters  = self.config['max_qp_iters']
        self.count = 0
        self.sent = False

    def run(self, thread):        
        t1 = time.time()
        # # # # # # # # # 
        # Read sensors  #
        # # # # # # # # # 
        q = self.joint_positions
        v = self.joint_velocities
        
        # # Add noise on the joint velocities if simulation 
        # if(self.RUN_SIM):
        #     v += 0.1*np.random.rand(self.nv)

        # When getting torque measurement from robot, do not forget to flip the sign
        if(not self.RUN_SIM):
            self.joint_torques_measured = -self.joint_torques_total  

        # # # # # # # # # 
        # # Update OCP  #
        # # # # # # # # # 
        time_to_circle  = int(thread.ti - self.T_CIRCLE)


        if(time_to_circle == 0): 
            print("Entering circle phase")
            self.position_at_contact_switch = self.robot.data.oMf[self.endeffFrameId].translation.copy()
            self.target_position[:,:] = self.position_at_contact_switch.copy()
            self.target_position_x = self.target_position[:,0] 
            self.target_position_y = self.target_position[:,1] 
            self.target_position_z = self.target_position[:,2]
        # If circle tracking phase enters the MPC horizon, start updating models from the end with tracking models      
        if(0 <= time_to_circle and time_to_circle <= self.NH_SIMU):
            self.TASK_PHASE = 1
            # If current time matches an OCP node 
            if(time_to_circle%self.OCP_TO_SIMU_CYCLES == 0):
                # Select IAM
                self.node_id_circle = self.Nh - int(time_to_circle/self.OCP_TO_SIMU_CYCLES)

        if(0 <= time_to_circle and time_to_circle%self.OCP_TO_SIMU_CYCLES == 0):
            # set position refs over current horizon
            ti  = int(time_to_circle/self.OCP_TO_SIMU_CYCLES)
            tf  = ti + self.Nh+1
            # Target in (x,y)  = circle trajectory + offset to start from current position instead of absolute target
            offset_xy = self.position_at_contact_switch- self.pdes
            self.target_position = self.target_position_traj[ti:tf,:] + offset_xy
            # Target in z is fixed to the anchor at switch (equals absolute target if RESET_ANCHOR = False)
            # No position tracking in z : redundant with zero activation weight on z
            # self.target_position[:,2]  = self.robot.data.oMf[self.endeffFrameId].translation[2].copy()
            # Record target signals
            self.target_position_x = self.target_position[:,0] 
            self.target_position_y = self.target_position[:,1] 
            self.target_position_z = self.target_position[:,2]

        # # # # # # #  
        # Solve OCP #
        # # # # # # #  
        # If planning cycle, fetch OCP solution
        self.t_child, self.t_child_1 = 0, 0
        if thread.ti % int(self.sim_to_plan_ratio) == 0:         
            # No pipe
            if(not USE_PIPE):
                self.count = 0
                self.us, self.xs, self.Ks, self.t_child, self.ddp_iter, self.t_child_1, self.cost, self.constraint_norm, self.gap_norm, self.qp_iters, self.kkt_norm = solveOCP(q, v, 
                                                                                                  self.ddp, 
                                                                                                  self.max_sqp_iter, 
                                                                                                  self.max_qp_iters,
                                                                                                  self.node_id_circle,
                                                                                                  self.target_position,
                                                                                                  self.TASK_PHASE)
            # With pipe
            else:
                if thread.ti != 0 and not self.sent:
                    self.is_plan_updated = False
                    self.parent_conn.send((q, v, self.target_position)) 
                    self.sent = True
            self.cumulative_cost += self.cost

        if USE_PIPE and self.parent_conn.poll() and not self.is_plan_updated:
            self.count = 0
            self.us, self.xs, self.Ks, self.t_child, self.ddp_iter, self.t_child_1, self.cost, self.constraint_norm, self.gap_norm, self.qp_iters, self.kkt_norm = self.parent_conn.recv()
            # record predictions here if necessary 
            self.sent = False

            # Increment planning counter
            self.nb_plan += 1
            self.is_plan_updated = True


        # # # # # # # # 
        # Send policy #
        # # # # # # # #
        # Linear interpolation of torque control input & desired state 
        ctr_index = int(self.count*self.ocp_to_sim_ratio)
        if ctr_index > self.Nh-1:
            self.tau_ff   = self.us[-1]     
            self.x_des    = self.xs[-1]  
            K = self.Ks[-1]
            print("DANGER")
        else:
            self.tau_ff   = self.us[ctr_index]     
            self.x_des    = self.xs[ctr_index+1]  
            K = self.Ks[ctr_index]

        # Riccati policy (optional) on (q,v) 
        if(self.config['RICCATI']):
            self.tau_riccati = K @ (self.x_des - np.concatenate([q, v]))
            self.tau  = self.tau_ff + self.tau_riccati
        else:
            self.tau_riccati = np.zeros(self.nv)
            self.tau = self.tau_ff
        
        self.count += 1

        # Compute gravity
        self.tau_gravity = pin.rnea(self.robot.model, self.robot.data, self.joint_positions, np.zeros(self.nv), np.zeros(self.nv))

        if(self.RUN_SIM == False):
            self.tau -= self.tau_gravity

        self.head.set_control('ctrl_joint_torques', self.tau)     


        pin.framesForwardKinematics(self.robot.model, self.robot.data, q)

        self.nb_ctrl += 1

        self.t_run = time.time() - t1