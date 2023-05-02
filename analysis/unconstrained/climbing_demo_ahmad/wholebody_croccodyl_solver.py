"""author: Ahmad Gazar"""


# Display the solution

import pathlib
import os
import sys
os.sys.path.insert(1, "/home/armand/CODE/gnms/python/sqp_ocp/solvers")


from gnms_cpp import GNMSCPP 
import crocoddyl
import pinocchio
import numpy as np

class WholeBodyDDPSolver:
    # constructor
    def __init__(self, model, centroidalTask=None, forceTask=None, MPC=False, WARM_START=True):
        self.RECEEDING_HORIZON = MPC
        # timing
        self.N_mpc = model.N_mpc
        self.N_traj = model.N
        self.dt = model.dt
        self.dt_ctrl = model.dt_ctrl
        self.N_interpol = int(model.dt/model.dt_ctrl)
        # whole-body croccodyl model
        self.whole_body_model = model    
        # initial condition and warm-start
        self.x0 = model.x0
        # extra tasks
        self.centroidal_task = centroidalTask
        self.force_task = forceTask
        # flags
        self.WARM_START = WARM_START
        self.RECEEDING_HORIZON = MPC
        # initialize ocp and create DDP solver
        self.__add_tracking_tasks(centroidalTask, forceTask)
        self.__init_ocp_and_solver()
        if MPC:
            self.warm_start_mpc(centroidalTask, forceTask)

    def __init_ocp_and_solver(self):
        wbd_model, N_mpc = self.whole_body_model,  self.N_mpc
        if self.RECEEDING_HORIZON:
           self.add_extended_horizon_mpc_models()
        else:
            wbd_model.terminal_model = self.add_terminal_cost()
            ocp = crocoddyl.ShootingProblem(self.x0, 
                                wbd_model.running_models, 
                                wbd_model.terminal_model)
            # self.solver = crocoddyl.SolverFDDP(ocp)
            # self.solver.setCallbacks([crocoddyl.CallbackLogger(),
                                    # crocoddyl.CallbackVerbose()])     

            self.solver = GNMSCPP(ocp, VERBOSE=True)
            self.solver.use_heuristic_ls = True

    
    def __add_tracking_tasks(self, centroidalTask, forceTask):
        if self.RECEEDING_HORIZON:
            N = self.N_mpc 
        else:
            N = self.N_traj
        running_models  = self.whole_body_model.running_models[:N]
        if forceTask is not None:
            self.add_force_tracking_cost(running_models, forceTask)
    
    def add_extended_horizon_mpc_models(self):
        N_mpc = self.N_mpc
        for _ in range(N_mpc):
            self.whole_body_model.running_models += [self.add_terminal_cost()]
        if self.force_task is not None:
            forceTaskTerminal_N = np.repeat(
                self.force_task[-1].reshape(1, self.force_task[-1].shape[0]), N_mpc, axis=0
                ) 
            self.add_force_tracking_cost(
                self.whole_body_model.running_models[self.N_mpc:], forceTaskTerminal_N 
            )

    def add_terminal_cost(self):
        wbd_model = self.whole_body_model
        final_contact_sequence = wbd_model.contact_sequence[-1]
        if wbd_model.rmodel.type == 'QUADRUPED':
            supportFeetIds = [wbd_model.lfFootId, wbd_model.rfFootId, 
                              wbd_model.lhFootId, wbd_model.rhFootId]
            feetPos = [final_contact_sequence[1].pose.translation, 
                       final_contact_sequence[0].pose.translation,
                       final_contact_sequence[3].pose.translation, 
                       final_contact_sequence[2].pose.translation]
        elif wbd_model.rmodel.type == 'HUMANOID':
            supportFeetIds = [wbd_model.lfFootId, wbd_model.rfFootId]   
            feetPos = [final_contact_sequence[1].pose.translation, 
                       final_contact_sequence[0].pose.translation]
        swingFootTask = []
        for i, p in zip(supportFeetIds, feetPos):
            swingFootTask += [[i, pinocchio.SE3(np.eye(3), p)]]
        terminalCostModel = wbd_model.createSwingFootModel(
            supportFeetIds, swingFootTask=swingFootTask, comTask=wbd_model.comRef
            )
        return terminalCostModel

    def add_com_task(self, diff_cost, com_des):
        state, nu = self.whole_body_model.state, self.whole_body_model.actuation.nu
        com_residual = crocoddyl.ResidualModelCoMPosition(state, com_des, nu)
        com_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
        com_track = crocoddyl.CostModelResidual(state, com_activation, com_residual)
        diff_cost.addCost("comTrack", com_track, self.whole_body_model.task_weights['comTrack'])
    
    def update_com_reference(self, dam, com_ref, TERMINAL=False):
        for _, cost in dam.costs.todict().items():
            # update CoM reference
            if isinstance(cost.cost.residual,  
                crocoddyl.libcrocoddyl_pywrap.ResidualModelCoMPosition):
                # print("updating com tracking reference at node ")
                cost.cost.residual.reference = com_ref
                return True
        return False    
    
    def update_centroidal_reference(self, dam, hg_ref):
        for _, cost in dam.costs.todict().items():
            # update CoM reference
            if isinstance(cost.cost.residual,  
                crocoddyl.libcrocoddyl_pywrap.ResidualModelCentroidalMomentum):
                # print("updating com tracking reference at node ")
                cost.cost.residual.reference = hg_ref
                return True
        return False    
        
    def add_centroidal_momentum_task(self, diff_cost, hg_des):
        wbd_model = self.whole_body_model
        state, nu = wbd_model.state, wbd_model.actuation.nu
        hg_residual = crocoddyl.ResidualModelCentroidalMomentum(state, hg_des, nu)
        hg_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1., 1., 1., 1.]))
        hg_track = crocoddyl.CostModelResidual(state, hg_activation, hg_residual)
        diff_cost.addCost("centroidalTrack", hg_track, wbd_model.task_weights['centroidalTrack'])  

    def add_centroidal_costs(self, iam_N, centroidal_ref_N):                              
        ## update running model
        for centroidal_ref_k, iam_k in zip(centroidal_ref_N, iam_N):
            dam_k = iam_k.differential.costs
            com_ref_k = centroidal_ref_k[:3]
            hg_ref_k = centroidal_ref_k[3:9]
            # update references if cost exists
            FOUND_COM_COST = self.update_com_reference(dam_k, com_ref_k)
            FOUND_HG_COST = self.update_centroidal_reference(dam_k, hg_ref_k)
            # create cost otherwise
            if not FOUND_COM_COST:
                # print("adding com tracking cost at node ", time_idx)
                self.add_com_task(dam_k, com_ref_k)
            if not FOUND_HG_COST:     
                self.add_centroidal_momentum_task(dam_k, hg_ref_k)
    
    def update_force_reference(self, dam, f_ref):
        wbd_model = self.whole_body_model
        rmodel, rdata = wbd_model.rmodel, wbd_model.rdata
        COST_REF_UPDATED = False
        for _, cost in dam.costs.todict().items():
            if isinstance(cost.cost.residual,  
                crocoddyl.libcrocoddyl_pywrap.ResidualModelContactForce):
                frame_idx = cost.cost.residual.id
                # print("updating force tracking reference for contact id ", frame_idx)
                pinocchio.framesForwardKinematics(rmodel, rdata, self.x0[:rmodel.nq])
                if frame_idx == wbd_model.rfFootId:
                    cost.cost.residual.reference = rdata.oMf[frame_idx].actInv(
                                                pinocchio.Force(f_ref[0:3], np.zeros(3))
                                                )
                elif frame_idx == wbd_model.lfFootId:
                    cost.cost.residual.reference = rdata.oMf[frame_idx].actInv(
                                                pinocchio.Force(f_ref[3:6], np.zeros(3))
                                                ) 
                elif frame_idx == wbd_model.rhFootId:
                    cost.cost.residual.reference = rdata.oMf[frame_idx].actInv(
                                                pinocchio.Force(f_ref[6:9], np.zeros(3)) 
                                                )
                elif frame_idx == wbd_model.lhFootId: 
                    cost.cost.residual.reference = rdata.oMf[frame_idx].actInv(
                                                pinocchio.Force(f_ref[9:12], np.zeros(3))
                                                )
                COST_REF_UPDATED = True      
        return COST_REF_UPDATED    

    def add_force_tasks(self, diff_cost, force_des, support_feet_ids):
        wbd_model = self.whole_body_model
        rmodel, rdata = wbd_model.rmodel, wbd_model.rdata 
        pinocchio.framesForwardKinematics(rmodel, rdata, self.x0[:rmodel.nq])
        state, nu = wbd_model.state, wbd_model.actuation.nu        
        forceTrackWeight = wbd_model.task_weights['contactForceTrack']
        if rmodel.foot_type == 'POINT_FOOT':
            nu_contact = 3
            linear_forces = force_des
        elif rmodel.type == 'HUMANOIND' and rmodel.foot_type == 'FLAT_FOOT':
            nu_contact = 6
            linear_forces = np.concatenate([force_des[2:5], force_des[8:11]])
        for frame_idx in support_feet_ids:
            # print("adding force tracking reference for contact id ", frame_idx)
            if frame_idx == wbd_model.rfFootId:
                spatial_force_des = rdata.oMf[frame_idx].actInv(
                    pinocchio.Force(linear_forces[0:3], np.zeros(3))
                    )
            elif frame_idx == wbd_model.lfFootId:
                spatial_force_des = rdata.oMf[frame_idx].actInv(
                    pinocchio.Force(linear_forces[3:6], np.zeros(3))
                    )
            elif frame_idx == wbd_model.rhFootId:
                spatial_force_des = rdata.oMf[frame_idx].actInv(
                    pinocchio.Force(linear_forces[6:9], np.zeros(3))
                    )
            elif frame_idx == wbd_model.lhFootId:
                spatial_force_des = rdata.oMf[frame_idx].actInv(
                    pinocchio.Force(linear_forces[9:12], np.zeros(3))
                    )
            force_activation_weights = np.array([1., 1., 1.])
            force_activation = crocoddyl.ActivationModelWeightedQuad(force_activation_weights)
            force_residual = crocoddyl.ResidualModelContactForce(state, frame_idx, 
                                                spatial_force_des, nu_contact, nu)
            force_track = crocoddyl.CostModelResidual(state, force_activation, force_residual)
            diff_cost.addCost(rmodel.frames[frame_idx].name +"contactForceTrack", 
                                                   force_track, forceTrackWeight)

    def add_force_tracking_cost(self, iam_N, force_ref_N):
        for force_ref_k, iam_k in zip(force_ref_N, iam_N):
            dam_k = iam_k.differential.costs
            support_foot_ids = []
            for _, cost in dam_k.costs.todict().items():
                if isinstance(cost.cost.residual,  
                    crocoddyl.libcrocoddyl_pywrap.ResidualModelContactFrictionCone):
                    support_foot_ids += [cost.cost.residual.id]
            FOUND_FORCE_COST = self.update_force_reference(dam_k, force_ref_k)        
            if not FOUND_FORCE_COST:
                self.add_force_tasks(dam_k, force_ref_k, support_foot_ids)
    
    def solve(self, x_warm_start=False, u_warm_start=False, max_iter=100):
        solver = self.solver
        if x_warm_start and u_warm_start:
            solver.solve(x_warm_start, u_warm_start)
        else:
            x0 = self.x0
            xs = [x0]*(solver.problem.T + 1)
            us = solver.problem.quasiStatic([x0]*solver.problem.T)
            solver.solve(xs, us, max_iter)    
        
    def get_solution_trajectories(self):
        xs, us, K = self.solver.xs, self.solver.us, self.solver.K
        rmodel, rdata = self.whole_body_model.rmodel, self.whole_body_model.rdata
        nq, nv, N = rmodel.nq, rmodel.nv, len(xs) 
        jointPos_sol = []
        jointVel_sol = []
        jointAcc_sol = []
        jointTorques_sol = []
        centroidal_sol = []
        gains = []
        x = []
        for time_idx in range (N):
            q, v = xs[time_idx][:nq], xs[time_idx][nq:]
            pinocchio.framesForwardKinematics(rmodel, rdata, q)
            pinocchio.computeCentroidalMomentum(rmodel, rdata, q, v)
            centroidal_sol += [
                np.concatenate(
                    [pinocchio.centerOfMass(rmodel, rdata, q, v), np.array(rdata.hg)]
                    )
                    ]
            jointPos_sol += [q]
            jointVel_sol += [v]
            x += [xs[time_idx]]
            if time_idx < N-1:
                jointAcc_sol +=  [self.solver.problem.runningDatas[time_idx].xnext[nq::]] 
                jointTorques_sol += [us[time_idx]]
                gains += [K[time_idx]]
        sol = {'x':x, 'centroidal':centroidal_sol, 'jointPos':jointPos_sol, 
                          'jointVel':jointVel_sol, 'jointAcc':jointAcc_sol, 
                            'jointTorques':jointTorques_sol, 'gains':gains}        
        return sol    
