import numpy as np
import pinocchio as pin 

import time

from classical_mpc.ocp import OptimalControlProblemClassical
from classical_mpc.data import MPCDataHandlerClassical #, DDPDataHandlerClassical
from core_mpc import path_utils, pin_utils, mpc_utils

from multiprocessing import Pipe, Process

USE_PIPE = False

def solveOCP(q, v, ddp, nb_iter, target_position):
    # Read state last measurement from parent process
    t = time.time()
    x = np.concatenate([q, v])
    # Update initial state + warm-start
    ddp.problem.x0 = x
    xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
    xs_init[0] = x
    us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
    # Update OCP 
    for i in range(ddp.problem.T):
        ddp.problem.runningModels[i].differential.costs.costs['translation'].cost.residual.reference = target_position
    ddp.problem.terminalModel.differential.costs.costs['translation'].cost.residual.reference = target_position
    problem_formulation_time = time.time()
    t_child_1 =  problem_formulation_time - t
    # Solve OCP 
    ddp.solve(xs_init, us_init, maxiter=nb_iter, isFeasible=False)
    solve_time = time.time()
    ddp_iter = ddp.iter
    t_child =  solve_time - problem_formulation_time
    return ddp.us, ddp.xs, ddp.K, t_child, ddp_iter, t_child_1




def rt_SolveOCP(child_conn, ddp, nb_iter, target_position):
    while True:
        # Read state last measurement from parent process
        q, v, target_position = child_conn.recv()
        t = time.time()
        x = np.concatenate([q, v])
        # Update initial state + warm-start
        ddp.problem.x0 = x
        xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
        xs_init[0] = x
        us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
        # Solve OCP 
        for i in range(ddp.problem.T):
            ddp.problem.runningModels[i].differential.costs.costs['translation'].cost.residual.reference = target_position
        ddp.problem.terminalModel.differential.costs.costs['translation'].cost.residual.reference = target_position
        problem_formulation_time = time.time()
        t_child_1 =  problem_formulation_time - t
        # Solve OCP 
        ddp.solve(xs_init, us_init, maxiter=nb_iter, isFeasible=False)
        # Send solution to parent process + riccati gains
        solve_time = time.time()
        ddp_iter = ddp.iter
        t_child =  solve_time - problem_formulation_time
        child_conn.send((ddp.us, ddp.xs, ddp.K, t_child, ddp_iter, t_child_1))        



class KukaReachSQP:

    def __init__(self, head, robot, config, run_sim, use_SQP):
        """
        Input:
            head        : thread head
            robot_model : pinocchio model
            config      : MPC config yaml file
            run_sim     : boolean sim or real
            use_SQP    : use SQP solver
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

        # Config
        self.config = config
        self.x0 = np.concatenate([self.joint_positions, self.joint_velocities]) 
        self.Nh = int(self.config['N_h'])
        self.dt_ocp  = self.config['dt']
        self.dt_plan = 1./self.config['plan_freq']
        self.dt_simu = 1./self.config['simu_freq']
        self.dt = 1./self.config['plan_freq']
        self.ocp_to_sim_ratio = 1. / ( self.config['simu_freq'] * self.config['dt'] )
        self.sim_to_plan_ratio = self.config['simu_freq']/self.config['plan_freq']
        self.ddp = OptimalControlProblemClassical(robot, self.config).initialize(self.x0, callbacks=False, USE_SQP=use_SQP)

        self.ug  = pin_utils.get_u_grav(self.x0[:self.robot.model.nq], self.robot.model, self.config['armature'])

        # Allocate MPC data
        self.us = self.ddp.us
        self.xs = self.ddp.xs
        self.Ks = self.ddp.K 
        self.x = self.xs[0]
        self.tau = self.us[0]
        self.K = self.Ks[0]

        self.nb_ctrl = 0
        self.nb_plan = 0
        self.nb_iter = self.config['maxiter'] 
        
        self.target2 = np.array([-0.4, -0.2, 0.8])
        self.target1 = np.asarray(self.config['frameTranslationRef'])
        self.target_position = self.target1
        self.current_target = 1
        self.Tswitch = 3000

 
    def warmup(self, thread):
        self.nb_iter = 100        
        self.ddp.xs = [self.x0 for i in range(self.config['N_h']+1)]
        self.ddp.us = [self.ug for i in range(self.config['N_h'])]
        self.is_plan_updated = False
        # USE pipe
        if(USE_PIPE):
            self.subp = Process(target=rt_SolveOCP, args=(self.child_conn, self.ddp, self.nb_iter, self.target_position))
            self.subp.start()
            # Read sensors and publish real state 
            self.parent_conn.send((self.joint_positions, self.joint_velocities, self.target_position))
            self.us, self.xs, self.Ks, self.t_child, self.ddp_iter, self.t_child_1  = self.parent_conn.recv()
        # NO pipe
        else:
            self.us, self.xs, self.Ks, self.t_child, self.ddp_iter, self.t_child_1 = solveOCP(self.joint_positions, 
                                                                                            self.joint_velocities, 
                                                                                            self.ddp, 
                                                                                            self.nb_iter,
                                                                                            self.target_position)
        self.check = 0
        self.nb_iter = self.config['maxiter']
        self.count = 0
        self.sent = False


    def run(self, thread):
        t1 = time.time()

        q = self.joint_positions
        v = self.joint_velocities

        # # # # # # #  
        # Solve OCP #
        # # # # # # #  
        # If planning cycle, fetch OCP solution
        self.t_child, self.t_child_1 = 0, 0
        if thread.ti % int(self.sim_to_plan_ratio) == 0:         
            # No pipe
            if(not USE_PIPE):
                self.count = 0
                self.us, self.xs, self.Ks, self.t_child, self.ddp_iter, self.t_child_1 = solveOCP(q, v, 
                                                                                                  self.ddp, 
                                                                                                  self.nb_iter,
                                                                                                  self.target_position)
            # With pipe
            else:
                if thread.ti != 0 and not self.sent:
                    self.is_plan_updated = False
                    self.parent_conn.send((q, v, self.target_position)) 
                    self.sent = True

        if USE_PIPE and self.parent_conn.poll() and not self.is_plan_updated:
            self.count = 0
            self.us, self.xs, self.Ks, self.t_child, self.ddp_iter, self.t_child_1 = self.parent_conn.recv()
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
