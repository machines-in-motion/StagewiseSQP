import solo_friction_utils
import example_robot_data
import crocoddyl
import pinocchio as pin
import numpy as np
from mim_robots.robot_loader import load_pinocchio_wrapper


def create_solo12_problem(MU):
    '''
    Create shooting problem for the double pendulum model
    '''

    pinRef        = pin.LOCAL_WORLD_ALIGNED

    ee_frame_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
    solo12 = example_robot_data.ROBOTS['solo12']()
    rmodel = solo12.robot.model
    rmodel.type = 'QUADRUPED'
    rmodel.foot_type = 'POINT_FOOT'
    rdata = rmodel.createData()

    # set contact frame_names and_indices
    lfFootId = rmodel.getFrameId(ee_frame_names[0])
    rfFootId = rmodel.getFrameId(ee_frame_names[1])
    lhFootId = rmodel.getFrameId(ee_frame_names[2])
    rhFootId = rmodel.getFrameId(ee_frame_names[3])


    q0 = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0] 
                    + 2 * [0.0, 0.8, -1.6] 
                    + 2 * [0.0, -0.8, 1.6] 
                    )

    x0 =  np.concatenate([q0, np.zeros(rmodel.nv)])

    pin.forwardKinematics(rmodel, rdata, q0)
    pin.updateFramePlacements(rmodel, rdata)
    rfFootPos0 = rdata.oMf[rfFootId].translation
    rhFootPos0 = rdata.oMf[rhFootId].translation
    lfFootPos0 = rdata.oMf[lfFootId].translation
    lhFootPos0 = rdata.oMf[lhFootId].translation 

    comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
    comRef[2] = pin.centerOfMass(rmodel, rdata, q0)[2].item() 

    supportFeetIds = [lfFootId, rfFootId, lhFootId, rhFootId]
    supportFeePos = [lfFootPos0, rfFootPos0, lhFootPos0, rhFootPos0]


    state = crocoddyl.StateMultibody(rmodel)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    nu = actuation.nu


    comDes = []

    N_ocp = 40
    dt = 0.02
    T = N_ocp * dt
    radius = 0.065
    for t in range(N_ocp+1):
        comDes_t = comRef.copy()
        w = (2 * np.pi) * 0.2 # / T
        # print(w * t * dt / 2 / np.pi)
        comDes_t[0] += radius * 1 # * np.sin(w * t * dt) 
        comDes_t[1] += radius * 0 # * (np.cos(w * t * dt) - 1)
        comDes += [comDes_t]

    running_models = []
    for i, t in enumerate(range(N_ocp+1)):
        contactModel = crocoddyl.ContactModelMultiple(state, nu)
        costModel = crocoddyl.CostModelSum(state, nu)

        # Add contact
        for i, frame_idx in enumerate(supportFeetIds):
            support_contact = crocoddyl.ContactModel3D(state, frame_idx, supportFeePos[i], pinRef, nu, np.array([0., 0.]))
            # print("contact name = ", rmodel.frames[frame_idx].name + "_contact")
            contactModel.addContact(rmodel.frames[frame_idx].name + "_contact", support_contact) 

        # Add state/control reg costs

        state_reg_weight, control_reg_weight = 1e-1, 1e-3

        freeFlyerQWeight = [0.]*3 + [500.]*3
        freeFlyerVWeight = [10.]*6
        legsQWeight = [0.01]*(rmodel.nv - 6)
        legsWWeights = [1.]*(rmodel.nv - 6)
        stateWeights = np.array(freeFlyerQWeight + legsQWeight + freeFlyerVWeight + legsWWeights)    


        stateResidual = crocoddyl.ResidualModelState(state, x0, nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(state, stateActivation, stateResidual)

        if t == N_ocp:
            costModel.addCost("stateReg", stateReg, state_reg_weight*dt)
        else:
            costModel.addCost("stateReg", stateReg, state_reg_weight)

        if t != N_ocp:
            ctrlResidual = crocoddyl.ResidualModelControl(state, nu)
            ctrlReg = crocoddyl.CostModelResidual(state, ctrlResidual)
            costModel.addCost("ctrlReg", ctrlReg, control_reg_weight)      


        # Add COM task
        com_residual = crocoddyl.ResidualModelCoMPosition(state, comDes[t], nu)
        com_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
        com_track = crocoddyl.CostModelResidual(state, com_activation, com_residual)
        if t == N_ocp:
            costModel.addCost("comTrack", com_track, 1e5)
        else:
            costModel.addCost("comTrack", com_track, 1e5)

        constraintModelManager = crocoddyl.ConstraintModelManager(state, actuation.nu)

        if(t != N_ocp):
            for frame_idx in supportFeetIds:
                name = rmodel.frames[frame_idx].name + "_contact"
                residualFriction = solo_friction_utils.ResidualFrictionCone(state, name, MU, actuation.nu)
                constraintFriction = crocoddyl.ConstraintModelResidual(state, residualFriction, np.array([0.]), np.array([np.inf]))
                constraintModelManager.addConstraint(name + "friction", constraintFriction)

        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, costModel, constraintModelManager, 0., True)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, dt)

        running_models += [model]

    # Create shooting problem
    ocp = crocoddyl.ShootingProblem(x0, running_models[:-1], running_models[-1])

    return ocp

def create_cartpole_problem(x0):
    '''
    Create shooting problem for Cartpole
    '''
    print("Create cartpole problem ...")
    # Creating the DAM for the cartpole
    cartpoleDAM = DifferentialActionModelCartpole()
    # Using NumDiff for computing the derivatives. We specify the
    # withGaussApprox=True to have approximation of the Hessian based on the
    # Jacobian of the cost residuals.
    cartpoleND = crocoddyl.DifferentialActionModelNumDiff(cartpoleDAM, True)
    # Getting the IAM using the simpletic Euler rule
    timeStep = 5e-2
    cartpoleIAM = crocoddyl.IntegratedActionModelEuler(cartpoleND, timeStep)
    # Creating the shooting problem
    T = 50
    terminalCartpole = DifferentialActionModelCartpole()
    terminalCartpoleDAM = crocoddyl.DifferentialActionModelNumDiff(terminalCartpole, True)
    terminalCartpoleIAM = crocoddyl.IntegratedActionModelEuler(terminalCartpoleDAM, 0.)
    terminalCartpole.costWeights[0] = 200
    terminalCartpole.costWeights[1] = 200
    terminalCartpole.costWeights[2] = 1.
    terminalCartpole.costWeights[3] = 0.1
    terminalCartpole.costWeights[4] = 0.01
    terminalCartpole.costWeights[5] = 0.0001
    pb = crocoddyl.ShootingProblem(x0, [cartpoleIAM] * T, terminalCartpoleIAM)
    return pb 

def create_kuka_problem(x0):
    '''
    Create shooting problem for kuka reaching task
    '''
    print("Create kuka problem ...")
    robot = load_pinocchio_wrapper("iiwa")
    model = robot.model
    # State and actuation model
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFull(state)

    # Create cost terms
    # Control regularization cost
    uResidual = crocoddyl.ResidualModelControlGrav(state)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    # State regularization cost
    xResidual = crocoddyl.ResidualModelState(state, x0)
    xRegCost = crocoddyl.CostModelResidual(state, xResidual)
    # endeff frame translation cost
    endeff_frame_id = model.getFrameId("contact")
    # ee velocity cost
    frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(
        state, endeff_frame_id, pin.Motion.Zero(), pin.LOCAL_WORLD_ALIGNED
    )
    frameVelocityCost = crocoddyl.CostModelResidual(state, frameVelocityResidual)
    # Create contraint on end-effector (small box around initial EE position)
    frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
        state, endeff_frame_id, np.zeros(3)
    )
    frameTranslationResidualCost = crocoddyl.ResidualModelFrameTranslation(
        state, endeff_frame_id, np.array([0.45, -0.2, 0.15])
    )
    frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidualCost)

    ee_contraint = crocoddyl.ConstraintModelResidual(
        state,
        frameTranslationResidual,
        np.array([0.45, -0.2, 0.15]),
        np.array([0.75, +0.2, 0.5]),
    )

    #  Constraint on frame velocity
    frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(
        state, endeff_frame_id, pin.Motion.Zero(), pin.LOCAL_WORLD_ALIGNED
    )

    # Create the running models
    runningModels = []
    dt = 1e-2
    T = 10
    for t in range(T + 1):
        runningCostModel = crocoddyl.CostModelSum(state)
        # Add costs
        runningCostModel.addCost("stateReg", xRegCost, 1e-1)
        runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
        runningCostModel.addCost("translation", frameTranslationCost, 1.)

        acc_refs = crocoddyl.ResidualModelJointAcceleration(state, np.zeros(7), actuation.nu)
        accCost = crocoddyl.CostModelResidual(state, acc_refs)

        acc_contraint = crocoddyl.ConstraintModelResidual(
            state,
            acc_refs,
            np.zeros(7),
            np.zeros(7),
        )

        runningCostModel.addCost("accCost", accCost, 1e-4)

        if t != T:
            runningCostModel.addCost("velocity", frameVelocityCost, 1e-3)
        else:
            runningCostModel.addCost("velocity", frameVelocityCost, 1e-1)
        # Define contraints
        constraints = crocoddyl.ConstraintModelManager(state, actuation.nu)
        if t == T: 
            constraints.addConstraint("ee_bound", ee_contraint)
        # if t == T-1:
        #     constraints.addConstraint("acc_contraint", acc_contraint)
        # Create Differential action model
        running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, runningCostModel, constraints
        )
        # Apply Euler integration
        running_model = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
        runningModels.append(running_model)        
    pb = crocoddyl.ShootingProblem(x0, runningModels[:-1], runningModels[-1])
    return pb

def create_humanoid_taichi_problem(target=np.array([0.4, 0, 1.2]), 
                                   JOINT_CONSTRAINT=False, 
                                   FORCE_COST=True,
                                   FORCE_CONSTRAINT=True):
    '''
    Create shooting problem for Talos taichi task
    '''
    # Load robot
    robot = example_robot_data.load("talos")
    rmodel = robot.model
    lims = rmodel.effortLimit
    # lims[19:] *= 0.5  # reduced artificially the torque limits
    rmodel.effortLimit = lims
    # Create data structures
    rdata = rmodel.createData()
    state = crocoddyl.StateMultibody(rmodel)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    # Set integration time
    DT = 5e-2
    T = 10
    # Initialize reference state, target and reference CoM
    rightFoot = "right_sole_link"
    leftFoot = "left_sole_link"
    endEffector = "gripper_left_joint"
    endEffectorId = rmodel.getFrameId(endEffector)
    rightFootId = rmodel.getFrameId(rightFoot)
    leftFootId = rmodel.getFrameId(leftFoot)
    q0 = rmodel.referenceConfigurations["half_sitting"]
    x0 = np.concatenate([q0, np.zeros(rmodel.nv)])
    pin.forwardKinematics(rmodel, rdata, q0)
    pin.updateFramePlacements(rmodel, rdata)
    rfPos0 = rdata.oMf[rightFootId].translation
    lfPos0 = rdata.oMf[leftFootId].translation
    comRef = (rfPos0 + lfPos0) / 2
    comRef[2] = pin.centerOfMass(rmodel, rdata, q0)[2].item()
    # Create two contact models used along the motion
    contactModel1Foot = crocoddyl.ContactModelMultiple(state, actuation.nu)
    contactModel2Feet = crocoddyl.ContactModelMultiple(state, actuation.nu)
    supportContactModelLeft = crocoddyl.ContactModel6D(
        state,
        leftFootId,
        pin.SE3.Identity(),
        pin.LOCAL_WORLD_ALIGNED,
        actuation.nu,
        np.array([0, 40]),
    )
    supportContactModelRight = crocoddyl.ContactModel6D(
        state,
        rightFootId,
        pin.SE3.Identity(),
        pin.LOCAL_WORLD_ALIGNED,
        actuation.nu,
        np.array([0, 40]),
    )
    contactModel1Foot.addContact(rightFoot + "_contact", supportContactModelRight)
    contactModel2Feet.addContact(leftFoot + "_contact", supportContactModelLeft)
    contactModel2Feet.addContact(rightFoot + "_contact", supportContactModelRight)
    # Cost for self-collision
    maxfloat = np.inf #sys.float_info.max !!! needs to be inf otherwise HPIPM would crash
    xlb = np.concatenate(
        [
            -maxfloat * np.ones(6),  # dimension of the SE(3) manifold
            rmodel.lowerPositionLimit[7:],
            -maxfloat * np.ones(state.nv),
        ]
    )
    xub = np.concatenate(
        [
            maxfloat * np.ones(6),  # dimension of the SE(3) manifold
            rmodel.upperPositionLimit[7:],
            maxfloat * np.ones(state.nv),
        ]
    )
    bounds = crocoddyl.ActivationBounds(xlb, xub, 1.0)
    xLimitResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
    xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
    limitCost = crocoddyl.CostModelResidual(state, xLimitActivation, xLimitResidual)
    # Cost for state and control
    xResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
    xActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [10] * state.nv) ** 2
    )
    uResidual = crocoddyl.ResidualModelControl(state, actuation.nu)
    xTActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [100] * state.nv) ** 2
    )
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    xRegTermCost = crocoddyl.CostModelResidual(state, xTActivation, xResidual)

    # Cost for target reaching: hand and foot
    handTrackingResidual = crocoddyl.ResidualModelFramePlacement(
        state, endEffectorId, pin.SE3(np.eye(3), target), actuation.nu
    )
    handTrackingActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([1] * 3 + [0.0001] * 3) ** 2
    )
    handTrackingCost = crocoddyl.CostModelResidual(
        state, handTrackingActivation, handTrackingResidual
    )

    footTrackingResidual = crocoddyl.ResidualModelFramePlacement(
        state, leftFootId, pin.SE3(np.eye(3), np.array([0.0, 0.4, 0.0])), actuation.nu
    )
    footTrackingActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([1, 1, 0.1] + [1.0] * 3) ** 2
    )
    footTrackingCost1 = crocoddyl.CostModelResidual(
        state, footTrackingActivation, footTrackingResidual
    )
    footTrackingResidual = crocoddyl.ResidualModelFramePlacement(
        state,
        leftFootId,
        pin.SE3(np.eye(3), np.array([0.3, 0.15, 0.35])),
        actuation.nu,
    )
    footTrackingCost2 = crocoddyl.CostModelResidual(
        state, footTrackingActivation, footTrackingResidual
    )
    # Cost for CoM reference
    # comResidual = crocoddyl.ResidualModelCoMPosition(state, comRef, actuation.nu)
    # comTrack = crocoddyl.CostModelResidual(state, comResidual)
    # Create cost model per each action model. We divide the motion in 3 phases plus its
    # terminal model.
    runningCostModel1 = crocoddyl.CostModelSum(state, actuation.nu)
    runningCostModel2 = crocoddyl.CostModelSum(state, actuation.nu)
    runningCostModel3 = crocoddyl.CostModelSum(state, actuation.nu)
    terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)
    # Then let's added the running and terminal cost functions
    runningCostModel1.addCost("gripperPose", handTrackingCost, 1e2)
    runningCostModel1.addCost("stateReg", xRegCost, 1e-3)
    runningCostModel1.addCost("ctrlReg", uRegCost, 1e-4)
    if not JOINT_CONSTRAINT:
        runningCostModel1.addCost("limitCost", limitCost, 1e3)
    runningCostModel2.addCost("gripperPose", handTrackingCost, 1e2)
    runningCostModel2.addCost("footPose", footTrackingCost1, 1e1)
    runningCostModel2.addCost("stateReg", xRegCost, 1e-3)
    runningCostModel2.addCost("ctrlReg", uRegCost, 1e-4)
    if not JOINT_CONSTRAINT:
        runningCostModel2.addCost("limitCost", limitCost, 1e3)
    runningCostModel3.addCost("gripperPose", handTrackingCost, 1e2)
    runningCostModel3.addCost("footPose", footTrackingCost2, 1e1)
    runningCostModel3.addCost("stateReg", xRegCost, 1e-3)
    runningCostModel3.addCost("ctrlReg", uRegCost, 1e-4)
    if not JOINT_CONSTRAINT:
        runningCostModel3.addCost("limitCost", limitCost, 1e3)
    terminalCostModel.addCost("gripperPose", handTrackingCost, 1e2)
    terminalCostModel.addCost("stateReg", xRegTermCost, 1e-3)
    if not JOINT_CONSTRAINT:
        terminalCostModel.addCost("limitCost", limitCost, 1e3)
    constraintModelManager = crocoddyl.ConstraintModelManager(state, actuation.nu)
    fref = pin.Force.Zero()
    ForceResidual = crocoddyl.ResidualModelContactForce(state, rightFootId, fref, 6, actuation.nu)
    if FORCE_COST:
        Forcecost = crocoddyl.CostModelResidual(state, ForceResidual)
        runningCostModel1.addCost("forcecost1", Forcecost, 1e-3)
        runningCostModel2.addCost("forcecost2", Forcecost, 1e-3)
        runningCostModel3.addCost("forcecost3", Forcecost, 1e-3)
    if FORCE_CONSTRAINT:
        constraintForce = crocoddyl.ConstraintModelResidual(state, ForceResidual, np.array([-np.inf, -np.inf, 0]*2), np.array([np.inf, np.inf, np.inf]*2))
        constraintModelManager.addConstraint("force", constraintForce)
    if JOINT_CONSTRAINT:
        constraintState = crocoddyl.ConstraintModelResidual(state, xLimitResidual, xlb, xub)
        constraintModelManager.addConstraint("state", constraintState)
    # Create the action model
    dmodelRunning1 = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contactModel2Feet, runningCostModel1, constraintModelManager, 0., True)
    dmodelRunning2 = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contactModel1Foot, runningCostModel2, constraintModelManager, 0., True)
    dmodelRunning3 = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contactModel1Foot, runningCostModel3, constraintModelManager, 0., True)
    dmodelTerminal = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contactModel1Foot, terminalCostModel
    )
    runningModel1 = crocoddyl.IntegratedActionModelEuler(dmodelRunning1, DT)
    runningModel2 = crocoddyl.IntegratedActionModelEuler(dmodelRunning2, DT)
    runningModel3 = crocoddyl.IntegratedActionModelEuler(dmodelRunning3, DT)
    terminalModel = crocoddyl.IntegratedActionModelEuler(dmodelTerminal, 0)
    # Problem definition
    pb = crocoddyl.ShootingProblem(
        x0, [runningModel1] * T + [runningModel2] * T + [runningModel3] * T, terminalModel
    )
    return pb

def create_quadrotor_problem(x0):
    hector = example_robot_data.load("hector")
    robot_model = hector.model

    target_pos = np.array([1.0, 0.0, 1.0])
    target_quat = pin.Quaternion(1.0, 0.0, 0.0, 0.0)

    state = crocoddyl.StateMultibody(robot_model)

    d_cog, cf, cm, u_lim, l_lim = 0.1525, 6.6e-5, 1e-6, 5.0, 0.1
    ps = [
        crocoddyl.Thruster(
            pin.SE3(np.eye(3), np.array([d_cog, 0, 0])),
            cm / cf,
            crocoddyl.ThrusterType.CCW,
        ),
        crocoddyl.Thruster(
            pin.SE3(np.eye(3), np.array([0, d_cog, 0])),
            cm / cf,
            crocoddyl.ThrusterType.CW,
        ),
        crocoddyl.Thruster(
            pin.SE3(np.eye(3), np.array([-d_cog, 0, 0])),
            cm / cf,
            crocoddyl.ThrusterType.CCW,
        ),
        crocoddyl.Thruster(
            pin.SE3(np.eye(3), np.array([0, -d_cog, 0])),
            cm / cf,
            crocoddyl.ThrusterType.CW,
        ),
    ]
    actuation = crocoddyl.ActuationModelFloatingBaseThrusters(state, ps)

    nu = actuation.nu
    runningCostModel = crocoddyl.CostModelSum(state, nu)
    terminalCostModel = crocoddyl.CostModelSum(state, nu)

    # Costs
    xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
    xActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([0.1] * 3 + [1000.0] * 3 + [1000.0] * robot_model.nv)
    )
    uResidual = crocoddyl.ResidualModelControl(state, nu)
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    goalTrackingResidual = crocoddyl.ResidualModelFramePlacement(
        state,
        robot_model.getFrameId("base_link"),
        pin.SE3(target_quat.matrix(), target_pos),
        nu,
    )
    goalTrackingCost = crocoddyl.CostModelResidual(state, goalTrackingResidual)
    runningCostModel.addCost("xReg", xRegCost, 1e-6)
    runningCostModel.addCost("uReg", uRegCost, 1e-6)
    runningCostModel.addCost("trackPose", goalTrackingCost, 1e-2)
    terminalCostModel.addCost("goalPose", goalTrackingCost, 3.0)

    constraints = crocoddyl.ConstraintModelManager(state, nu)

    uResidual = crocoddyl.ResidualModelControl(state, nu)
    u_contraint = crocoddyl.ConstraintModelResidual(
        state,
        uResidual,
        np.array([l_lim, l_lim, l_lim, l_lim]),
        np.array([u_lim, u_lim, u_lim, u_lim]),
    )
       
    constraints.addConstraint("u_contraint", u_contraint)


    dt = 3e-2
    runningModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, runningCostModel , constraints
        ),
        dt,
    )
    terminalModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, terminalCostModel
        ),
        dt,
    )

    # Creating the shooting problem and the BoxDDP solver
    T = 200 #33
    print(hector.q0)
    problem = crocoddyl.ShootingProblem(
        x0, [runningModel] * T, terminalModel
    )

    return problem