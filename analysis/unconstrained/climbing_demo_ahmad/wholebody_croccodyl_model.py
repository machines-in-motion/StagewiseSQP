"""author: Ahmad Gazar"""
import numpy as np
import crocoddyl
import pinocchio 

class WholeBodyModel:
    def __init__(self, conf):
        self.dt = conf.dt
        self.dt_ctrl = conf.dt_ctrl
        self.rmodel = conf.rmodel
        self.rdata = conf.rmodel.createData()
        self.ee_frame_names = conf.ee_frame_names 
        self.gait = conf.gait
        self.contact_sequence = conf.contact_sequence
        self.gait_templates = conf.gait_templates 
        self.task_weights = conf.whole_body_task_weights
        self.state_reg_weights = conf.wbd_state_reg_weights
        # Defining the friction coefficient and normal
        self.postImpact = None
        self.mu = conf.mu
        self.N = conf.N
        self.N_mpc = conf.N_mpc_wbd
        self.Rsurf = np.eye(3)
        if conf.rmodel.foot_type == 'FLAT_FOOT':
            self.foot_size = np.array([conf.lxp-conf.lxn,
                                       conf.lyp-conf.lyn])
        self.__initialize_robot(conf.q0)
        self.__set_contact_frame_names_and_indices()
        self.__fill_ocp_models()

    def __fill_ocp_models(self):
        if self.gait['type'] == 'TROT':
            self.create_trot_models()
        
    def __set_contact_frame_names_and_indices(self):
        ee_frame_names = self.ee_frame_names
        rmodel = self.rmodel 
        if self.rmodel.type == 'QUADRUPED':
            self.lfFootId = rmodel.getFrameId(ee_frame_names[0])
            self.rfFootId = rmodel.getFrameId(ee_frame_names[1])
            self.lhFootId = rmodel.getFrameId(ee_frame_names[2])
            self.rhFootId = rmodel.getFrameId(ee_frame_names[3])
        elif rmodel.type == 'HUMANOID':
            self.lfFootId = rmodel.getFrameId(ee_frame_names[0])
            self.rfFootId = rmodel.getFrameId(ee_frame_names[1])

    def __initialize_robot(self, q0):
        self.rmodel.defaultState = np.concatenate([q0, np.zeros(self.rmodel.nv)])
        self.x0 = self.rmodel.defaultState
        # create croccodyl state and controls
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)

    def add_swing_feet_tracking_costs(self, cost, swing_feet_tasks):
        swingFootPosWeight = self.task_weights['swingFoot']['preImpact']['position']
        swingFootVelWeight = self.task_weights['swingFoot']['preImpact']['velocity']
        state, nu = self.state, self.actuation.nu
        for task in swing_feet_tasks:
            if self.rmodel.foot_type == 'POINT_FOOT':
                frame_pose_residual = crocoddyl.ResidualModelFrameTranslation(state,
                                                        task[0], task[1].translation, nu)
            elif self.rmodel.foot_type == 'FLAT_FOOT':
                frame_pose_residual = crocoddyl.ResidualModelFramePlacement(state, task[0], task[1], nu)
            verticalFootVelResidual = crocoddyl.ResidualModelFrameVelocity(state, task[0],
                    pinocchio.Motion.Zero(), pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED, nu
                    )
            verticalFootVelAct = crocoddyl.ActivationModelWeightedQuad(np.array([0, 0, 1, 0, 0, 0]))
            verticalFootVelCost = crocoddyl.CostModelResidual(state, verticalFootVelAct, 
                                                                verticalFootVelResidual)
            cost.addCost(self.rmodel.frames[task[0]].name+  "__footVelTrack", verticalFootVelCost, 
                                                                                swingFootVelWeight)                               
            foot_track = crocoddyl.CostModelResidual(state, frame_pose_residual)
            cost.addCost(self.rmodel.frames[task[0]].name + "_footPosTrack", foot_track, 
                                                                     swingFootPosWeight)
    
    def add_pseudo_impact_costs(self, cost, swing_feet_tasks):
        state, nu = self.state, self.actuation.nu
        footPosImpactWeight = self.task_weights['swingFoot']['impact']['position']
        footVelImpactweight = self.task_weights['swingFoot']['impact']['velocity']
        for task in swing_feet_tasks:
            frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, 
                                                    task[0], task[1].translation, nu)
            frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(state, task[0], 
                                           pinocchio.Motion.Zero(), pinocchio.LOCAL, nu)                          
            footPosImpactCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)
            footVelImpactCost = crocoddyl.CostModelResidual(state, frameVelocityResidual)
            cost.addCost(self.rmodel.frames[task[0]].name + "_footPosImpact",
                                           footPosImpactCost, footPosImpactWeight)
            cost.addCost(self.rmodel.frames[task[0]].name + "_footVelImpact", 
                                           footVelImpactCost, footVelImpactweight)
        if self.rmodel.foot_type == 'FLAT_FOOT':
            # keep feet horizontal at the time of impact
            for task in swing_feet_tasks:
                footRotImpactWeight = self.task_weights['swingFoot']['impact']['orientation']
                frameRotResidual = crocoddyl.ResidualModelFrameRotation(state, task[0], np.eye(3), nu)
                frameRotAct = crocoddyl.ActivationModelWeightedQuad(np.array([1, 1, 0]))
                footRotImpactCost = crocoddyl.CostModelResidual(state,frameRotAct , frameRotResidual) 
                cost.addCost(self.rmodel.frames[task[0]].name + "_footRotImpact",
                                            footRotImpactCost, footRotImpactWeight)                                                                                     

    def add_support_contact_costs(self, contact_model, cost, support_feet_ids):
        state, nu = self.state, self.actuation.nu
        rmodel = self.rmodel
        frictionConeWeight = self.task_weights['frictionCone']
        # check if it's a post-impact knot
        if self.postImpact is not None:
            self.add_pseudo_impact_costs(cost, self.postImpact)
        for frame_idx in support_feet_ids:
            R_cone_local = self.rdata.oMf[frame_idx].rotation.T.dot(self.Rsurf)
            if rmodel.foot_type == 'POINT_FOOT': 
                support_contact = crocoddyl.ContactModel3D(state, frame_idx, np.array([0., 0., 0.]), 
                                                                           nu, np.array([0., 50.]))
                cone = crocoddyl.FrictionCone(R_cone_local, self.mu, 4, True)
                cone_residual = crocoddyl.ResidualModelContactFrictionCone(state, frame_idx, cone, nu)
            elif rmodel.foot_type == 'FLAT_FOOT':
                # friction cone
                support_contact = crocoddyl.ContactModel6D(state, frame_idx, pinocchio.SE3.Identity(),
                                                                              nu, np.array([0., 50.]))
                cone = crocoddyl.WrenchCone(self.Rsurf, self.mu, np.array([self.foot_size[0], self.foot_size[1]]))
                cone_residual = crocoddyl.ResidualModelContactWrenchCone(state, frame_idx, cone, nu)
                # CoP
                cop_box = crocoddyl.CoPSupport(self.Rsurf, self.foot_size)
                cop_residual = crocoddyl.ResidualModelContactCoPPosition(state, frame_idx, cop_box, nu)
                cop_activation = crocoddyl.ActivationModelQuadraticBarrier(
                    crocoddyl.ActivationBounds(cop_box.lb, cop_box.ub)
                    )
                cop = crocoddyl.CostModelResidual(state, cop_activation, cop_residual)
                cost.addCost(rmodel.frames[frame_idx].name + "_cop", cop, self.task_weights['cop'])
            contact_model.addContact(rmodel.frames[frame_idx].name + "_contact", support_contact) 
            cone_activation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
                )
            friction_cone = crocoddyl.CostModelResidual(state, cone_activation, cone_residual)
            cost.addCost(rmodel.frames[frame_idx].name + "_frictionCone", friction_cone, frictionConeWeight)
    
    def add_com_position_tracking_cost(self, cost, com_des):    
        com_residual = crocoddyl.ResidualModelCoMPosition(self.state, com_des, self.actuation.nu)
        com_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
        com_track = crocoddyl.CostModelResidual(self.state, com_activation, com_residual)
        cost.addCost("comTrack", com_track, self.task_weights['comTrack'])

    def add_stat_ctrl_reg_costs(self, cost):
        nu = self.actuation.nu 
        stateWeights = self.state_reg_weights
        if self.postImpact is not None:
            state_reg_weight, control_reg_weight = self.task_weights['stateReg']['impact'],\
                                                    self.task_weights['ctrlReg']['impact']
            self.postImpact = None                                        
        else:
            state_reg_weight, control_reg_weight = self.task_weights['stateReg']['stance'],\
                                                    self.task_weights['ctrlReg']['stance']
        state_bounds_weight = self.task_weights['stateBounds']
        # state regularization cost
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
        cost.addCost("stateReg", stateReg, state_reg_weight)
        # state bounds cost
        lb = np.concatenate([self.state.lb[1:self.state.nv + 1], self.state.lb[-self.state.nv:]])
        ub = np.concatenate([self.state.ub[1:self.state.nv + 1], self.state.ub[-self.state.nv:]])
        stateBoundsResidual = crocoddyl.ResidualModelState(self.state, nu)
        stateBoundsActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub))
        stateBounds = crocoddyl.CostModelResidual(self.state, stateBoundsActivation, stateBoundsResidual)
        cost.addCost("stateBounds", stateBounds, state_bounds_weight)
        # control regularization cost
        ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        cost.addCost("ctrlReg", ctrlReg, control_reg_weight)                  
        
    def create_trot_models(self):
        # Compute the current foot positions
        x0 = self.rmodel.defaultState
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation 
        self.comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        self.comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2].item()
        self.time_idx = 0
        # Defining the action models along the time instances
        loco3dModel = []
        for gait in self.gait_templates:
            for phase in gait:
                if phase == 'doubleSupport':
                    loco3dModel += self.createDoubleSupportFootstepModels([lfFootPos0, rfFootPos0, 
                                                                          lhFootPos0, rhFootPos0])
                elif phase == 'rflhStep':
                    loco3dModel += self.createSingleSupportFootstepModels([rfFootPos0, lhFootPos0], 
                                    [self.lfFootId, self.rhFootId], [self.rfFootId, self.lhFootId])
                elif phase == 'lfrhStep':
                    loco3dModel += self.createSingleSupportFootstepModels([lfFootPos0, rhFootPos0], 
                                    [self.rfFootId, self.lhFootId], [self.lfFootId, self.rhFootId])
        self.running_models = loco3dModel

    def createDoubleSupportFootstepModels(self, feetPos):
        if self.rmodel.type == 'QUADRUPED':
            supportFeetIds = [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId]
        elif self.rmodel.type == 'HUMANOID':
            supportFeetIds = [self.lfFootId, self.rfFootId]
        supportKnots = self.gait['supportKnots']
        doubleSupportModel = []
        for _ in range(supportKnots):
            swingFootTask = []
            for i, p in zip(supportFeetIds, feetPos):
                swingFootTask += [[i, pinocchio.SE3(np.eye(3), p)]]
          
            doubleSupportModel += [self.createSwingFootModel(supportFeetIds, swingFootTask=swingFootTask)]               
        return doubleSupportModel

    def createSingleSupportFootstepModels(self, feetPos0, supportFootIds, swingFootIds):
        numLegs = len(supportFootIds) + len(swingFootIds)
        comPercentage = float(len(swingFootIds)) / numLegs
        stepLength, stepHeight = self.gait['stepLength'], self.gait['stepHeight']
        numKnots = self.gait['stepKnots'] 
        # Action models for the foot swing
        footSwingModel = []
        for k in range(numKnots):
            swingFootTask = []
            for i, p in zip(swingFootIds, feetPos0):
                # Defining a foot swing task given the step length
                # resKnot = numKnots % 2
                phKnots = numKnots - 5
                if k < phKnots:
                    dp = np.array([stepLength * (k + 1) / numKnots, 0., 1.75*stepHeight * k / phKnots])
                elif k == phKnots:
                    dp = np.array([stepLength * (k + 1) / numKnots, 0., 1.75*stepHeight])
                else:
                    dp = np.array(
                        [stepLength * (k + 1) / numKnots, 0., 1.75*stepHeight * (1 - float(k - phKnots) / phKnots)])
                tref = p + dp
                swingFootTask += [[i, pinocchio.SE3(np.eye(3), tref)]]    
            comTask = np.array([stepLength * (k + 1) / numKnots, 0., stepHeight * (k + 1)/numKnots ]) * comPercentage + self.comRef
            # postImpact = False 
            if k == numKnots-1:
                self.postImpact = swingFootTask
            else:
                self.postImpact = None    
            footSwingModel += [
                self.createSwingFootModel(supportFootIds, comTask=comTask, swingFootTask=swingFootTask)
                    ]
        # Updating the current foot position for next step
        self.comRef += [stepLength * comPercentage, 0., 0.5*stepHeight]
        for p in feetPos0:
            p += [stepLength, 0., stepHeight] #0.5*ste 
        return footSwingModel 

    def createSwingFootModel(self, supportFootIds, comTask=None, swingFootTask=None):
        # Creating a multi-contact model
        nu = self.actuation.nu
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        if isinstance(comTask, np.ndarray):
            self.add_com_position_tracking_cost(costModel, comTask)
        if swingFootTask is not None:
            self.add_swing_feet_tracking_costs(costModel, swingFootTask)
        self.add_support_contact_costs(contactModel, costModel, supportFootIds)
        self.add_stat_ctrl_reg_costs(costModel)
        # Creating the action model for the KKT dynamics with simpletic Euler integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, self.actuation, contactModel, costModel, 0., True
            )
        model = crocoddyl.IntegratedActionModelEuler(dmodel, self.dt)
        return model

    def get_solution_trajectories(self, solver):
        xs, us, K = solver.xs, solver.us, solver.K
        rmodel, rdata = self.rmodel, self.rdata
        nq, nv, N = rmodel.nq, rmodel.nv, len(xs) 
        jointPos_sol = np.zeros((N, nq))
        jointVel_sol = np.zeros((N, nv))
        jointAcc_sol = np.zeros((N, nv))
        jointTorques_sol = np.zeros((N-1, nv-6))
        centroidal_sol = np.zeros((N, 9))
        gains = np.zeros((N-1, K[0].shape[0], K[0].shape[1]))
        for time_idx in range (N):
            q, v = xs[time_idx][:nq], xs[time_idx][nq:]
            pinocchio.framesForwardKinematics(rmodel, rdata, q)
            pinocchio.computeCentroidalMomentum(rmodel, rdata, q, v)
            centroidal_sol[time_idx, :3] = pinocchio.centerOfMass(rmodel, rdata, q, v)
            centroidal_sol[time_idx, 3:9] = np.array(rdata.hg)
            jointPos_sol[time_idx, :] = q
            jointVel_sol[time_idx, :] = v
            if time_idx < N-1:
                jointAcc_sol[time_idx+1, :]= solver.problem.runningDatas[time_idx].xnext[nq::] 
                jointTorques_sol[time_idx, :] = us[time_idx]
                gains[time_idx, :,:] = K[time_idx]
        sol = {'centroidal':centroidal_sol, 'jointPos':jointPos_sol, 
               'jointVel':jointVel_sol, 'jointAcc':jointAcc_sol, 
               'jointTorques':jointTorques_sol, 'gains':gains}        
        return sol    


