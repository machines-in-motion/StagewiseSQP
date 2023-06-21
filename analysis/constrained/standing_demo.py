import crocoddyl
import pinocchio
import numpy as np
import example_robot_data 
from robot_properties_solo.solo12wrapper import Solo12Config
import pinocchio as pin
import standing_utils
import sobec

# from friction_cone import FrictionConeConstraint, Force3DConstraintModel
import sys

pinRef        = pin.LOCAL_WORLD_ALIGNED
FORCE_CSTR    = False
FRICTION_CSTR = False
PLOT = True

robot_name = 'solo12'
ee_frame_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
solo12 = example_robot_data.ROBOTS[robot_name]()
rmodel = solo12.robot.model
rmodel.type = 'QUADRUPED'
rmodel.foot_type = 'POINT_FOOT'
rdata = rmodel.createData()

# set contact frame_names and_indices
lfFootId = rmodel.getFrameId(ee_frame_names[0])
rfFootId = rmodel.getFrameId(ee_frame_names[1])
lhFootId = rmodel.getFrameId(ee_frame_names[2])
rhFootId = rmodel.getFrameId(ee_frame_names[3])


q0 = np.array(Solo12Config.initial_configuration.copy())
q0[0] = 0.0

x0 =  np.concatenate([q0, np.zeros(rmodel.nv)])

pinocchio.forwardKinematics(rmodel, rdata, q0)
pinocchio.updateFramePlacements(rmodel, rdata)
rfFootPos0 = rdata.oMf[rfFootId].translation
rhFootPos0 = rdata.oMf[rhFootId].translation
lfFootPos0 = rdata.oMf[lfFootId].translation
lhFootPos0 = rdata.oMf[lhFootId].translation 

comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
comRef[2] = pinocchio.centerOfMass(rmodel, rdata, q0)[2].item() 

supportFeetIds = [lfFootId, rfFootId, lhFootId, rhFootId]
supportFeePos = [lfFootPos0, rfFootPos0, lhFootPos0, rhFootPos0]


state = crocoddyl.StateMultibody(rmodel)
actuation = crocoddyl.ActuationModelFloatingBase(state)
nu = actuation.nu


comDes = []

N_ocp = 100 #100
dt = 0.01
T = N_ocp * dt
radius = 0.1
for t in range(N_ocp+1):
    comDes_t = comRef.copy()
    w = (2 * np.pi) / T
    comDes_t[0] += radius * np.sin(w * t * dt) 
    comDes_t[1] += radius * (np.cos(w * t * dt) - 1)
    comDes += [comDes_t]

running_models = []
constraintModels = []
for t in range(N_ocp+1):
    contactModel = sobec.ContactModelMultiple(state, nu)
    costModel = crocoddyl.CostModelSum(state, nu)

    # Add contact
    for frame_idx in supportFeetIds:
        support_contact = sobec.ContactModel3D(state, frame_idx, np.array([0., 0., 0.]), nu, np.array([0., 50.]), pinRef)
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

    # TO-DO: Add state bounds cost with hard constraint?

    # lb = np.concatenate([state.lb[1:state.nv + 1], state.lb[-self.state.nv:]])
    # ub = np.concatenate([state.ub[1:state.nv + 1], state.ub[-self.state.nv:]])
    # stateBoundsResidual = crocoddyl.ResidualModelState(state, nu)
    # stateBoundsActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub))
    # stateBounds = crocoddyl.CostModelResidual(state, stateBoundsActivation, stateBoundsResidual)
    # cost.addCost("stateBounds", stateBounds, state_bounds_weight)


    # Add contact force constraint >= 0 & friction cone 
    clip_force_min = np.array([-np.inf, -np.inf, 5])#, -np.inf, -np.inf, -np.inf])
    clip_force_max = np.array([np.inf, np.inf, np.inf]) # np.inf, np.inf, np.inf])
    forceConstraintModels = []
    frictionConstraintModels = []
    n_cstr = 0
    for frame_idx in supportFeetIds:
        if(FORCE_CSTR):
            # force_cstr = crocoddyl.ContactForceConstraintModel3D(state, actuation.nu, frame_idx, clip_force_min, clip_force_max, rmodel.frames[frame_idx].name+"_forceConstraint", pinRef)
            # force_cstr.contact_dynamics_ref = pinRef   
            force_cstr = Force3DConstraintModel(state, clip_force_min, clip_force_max, 3, state.nx, actuation.nu) 
            forceConstraintModels.append(force_cstr)
            n_cstr += force_cstr.nc
        # if(FRICTION_CSTR):
        #     friction_cstr = FrictionConeConstraint(state, 0.8, 1, actuation.nu, frame_idx, pinRef)
        #     frictionConstraintModels.append(friction_cstr)
        #     n_cstr += friction_cstr.nc
    
    # Create constraint model stack for the current node
    # runningConstraintModel = crocoddyl.ConstraintStack([fc for fc in forceConstraintModels]+[fc for fc in frictionConstraintModels], state, n_cstr, actuation.nu, 'runningConstraintModel')
    # runningConstraintModel = crocoddyl.ConstraintStack([fc for fc in forceConstraintModels], state, n_cstr, actuation.nu, 'runningConstraintModel')

    # clip_state_max = np.array([np.inf]*(state.nx))
    # clip_state_min = -np.array([np.inf]*(state.nx))
    # statemodel = crocoddyl.StateConstraintModel(state, actuation.nu, clip_state_min, clip_state_max, 'stateConstraint')
    # runningConstraintModel = crocoddyl.ConstraintStack([statemodel], state, statemodel.nc, actuation.nu, 'runningConstraintModel')

    # Append the constraint model stack to the list of constraint models
    # if( t == 0 or t == N_ocp):
    #     constraintModels += [crocoddyl.NoConstraintModel(state, actuation.nu, "noCstr")]
    # else:
    #     constraintModels += [runningConstraintModel] 

    dmodel = sobec.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, costModel, 0., True)
    model = crocoddyl.IntegratedActionModelEuler(dmodel, dt)

    running_models += [model]

# Create shooting problem
ocp = crocoddyl.ShootingProblem(x0, running_models[:-1], running_models[-1])

# Create solver , warm-start and solve
solver = crocoddyl.SolverFDDP(ocp)
# solver = crocoddyl.SolverFADMM(ocp, constraintModels)
# solver = crocoddyl.SolverPROXQP(ocp, constraintModels)
# solver.with_callbacks = True
# solver.use_filter_ls = True
# solver.filter_size = 200
# solver.termination_tolerance = 1e-3
# solver.eps_abs = 1e-9
# solver.eps_rel = 0.
# solver.max_qp_iters = 1000
# solver.KKT = True
  
# solver.setCallbacks([crocoddyl.CallbackLogger(),
#                      crocoddyl.CallbackVerbose()])    
xs = [x0]*(solver.problem.T + 1)
us = [np.zeros(actuation.nu)]*solver.problem.T #solver.us #solver.problem.quasiStatic([x0]*solver.problem.T)
max_iter = 200
solver.solve(xs, us, max_iter)   


solution = standing_utils.get_solution_trajectories(solver, rmodel, rdata, supportFeetIds)
q_sol = solution['jointPos']
centroidal_sol = solution['centroidal']


# Plot results
import matplotlib.pyplot as plt
if(PLOT):

    comDes = np.array(comDes)
    centroidal_sol = np.array(centroidal_sol)
    plt.figure()
    plt.plot(comDes[:, 0], comDes[:, 1], "--", label="reference")
    plt.plot(centroidal_sol[:, 0], centroidal_sol[:, 1], label="solution")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("COM trajectory")



    for frame_idx in supportFeetIds:
        force_list = []
        ct_frame_name = rmodel.frames[frame_idx].name + "_contact"

    time_lin = np.linspace(0, T, solver.problem.T)
    fig, axs = plt.subplots(4, 3, constrained_layout=True)
    for i, frame_idx in enumerate(supportFeetIds):
        ct_frame_name = rmodel.frames[frame_idx].name + "_contact"
        forces = np.array(solution[ct_frame_name])
        axs[i, 0].plot(time_lin, forces[:, 0], label="Fx")
        axs[i, 1].plot(time_lin, forces[:, 1], label="Fy")
        axs[i, 2].plot(time_lin, forces[:, 2], label="Fz")
        axs[i, 0].grid()
        axs[i, 1].grid()
        axs[i, 2].grid()
        axs[i, 0].set_ylabel(ct_frame_name)
    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[0, 2].legend()

    axs[3, 0].set_xlabel(r"$F_x$")
    axs[3, 1].set_xlabel(r"$F_y$")
    axs[3, 2].set_xlabel(r"$F_z$")
    fig.suptitle('Force', fontsize=16)



    fig, axs = plt.subplots(actuation.nu ,1)
    torques = np.array(solution["jointTorques"])
    for i in range(actuation.nu):
        axs[i].plot(time_lin, torques[:, i])
        axs[i].set_ylabel("$\\tau_{%s}$"%i)
        axs[i].grid()
    fig.suptitle('Control input', fontsize=16)



    time_lin = np.linspace(0, T, solver.problem.T+1)

    jointPos = np.array(solution["jointPos"])
    fig, axs = plt.subplots(rmodel.nq ,1)
    for i in range(rmodel.nq):
        axs[i].plot(time_lin, jointPos[:, i])
        axs[i].set_ylabel("$q_{%s}$"%i)
        axs[i].grid()
    fig.suptitle('Joint position', fontsize=16)

    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    for i in range(3):
        axs[i].plot(time_lin, comDes[:, i], "--", label="reference")
        axs[i].plot(time_lin, centroidal_sol[:, i], label="solution")
        axs[i].grid()
    axs[0].set_ylabel("$COM_{x}$")
    axs[1].set_ylabel("$COM_{y}$")
    axs[2].set_ylabel("$COM_{z}$")
    plt.legend()    

    plt.title("COM trajectory")

    # plt.show()


from meshcat.animation import Animation
import meshcat.transformations as tf    
# create robot
robot = Solo12Config.buildRobotWrapper()
# load robot in meshcat viewer
viz = pin.visualize.MeshcatVisualizer(
robot.model, robot.collision_model, robot.visual_model)
try:
    viz.initViewer(open=True)
except ImportError as err:
    print(err)
    sys.exit(0)
viz.loadViewerModel()


# angle = 0.0  # Initial angle
# rotation_speed = 0.05  # Speed of rotation (adjust as needed)

cam_pose = tf.translation_matrix([-3.5, 0, 0.])  # Example camera position
cam_pose[:3, :3] = tf.euler_matrix(0.0, 0.0, np.pi/6)[:3, :3]  # Example camera orientation
viz.viewer["/Cameras"].set_transform(cam_pose)



# add contact surfaces
step_adjustment_bound = 0.07                         
s = 0.5*step_adjustment_bound

for contact_idx, contactLoc in enumerate(supportFeePos):
    t = contactLoc
    # debris box
    standing_utils.addViewerBox(
        viz, 'world/debris'+str(contact_idx), 
        2*s, 2*s, 0., [1., .2, .2, .5]
        )
    standing_utils.applyViewerConfiguration(
        viz, 'world/debris'+str(contact_idx), 
        [t[0], t[1], t[2]-0.017, 1, 0, 0, 0]
        )
    standing_utils.applyViewerConfiguration(
        viz, 'world/debris_center'+str(contact_idx), 
        [t[0], t[1], t[2]-0.017, 1, 0, 0, 0]
        ) 


arrow1 = standing_utils.Arrow(viz.viewer, "force_1", location=[0,0,0], vector=[0,0,0.01], length_scale=0.05)
arrow2 = standing_utils.Arrow(viz.viewer, "force_2", location=[0,0,0], vector=[0,0,0.01], length_scale=0.05)
arrow3 = standing_utils.Arrow(viz.viewer, "force_3", location=[0,0,0], vector=[0,0,0.01], length_scale=0.05)
arrow4 = standing_utils.Arrow(viz.viewer, "force_4", location=[0,0,0], vector=[0,0,0.01], length_scale=0.05)

cone1 = standing_utils.Cone(viz.viewer, "friction_cone_1", location=supportFeePos[0], mu=0.5)
cone2 = standing_utils.Cone(viz.viewer, "friction_cone_2", location=supportFeePos[1], mu=0.5)
cone3 = standing_utils.Cone(viz.viewer, "friction_cone_3", location=supportFeePos[2], mu=0.5)
cone4 = standing_utils.Cone(viz.viewer, "friction_cone_4", location=supportFeePos[3], mu=0.5)

arrows = [arrow1, arrow2, arrow3, arrow4]
forces = []

for i, contactLoc in enumerate(supportFeePos):
    ct_frame_name = rmodel.frames[supportFeetIds[i]].name + "_contact"
    forces.append(np.array(solution[ct_frame_name])[:, :3])
    arrows[i].set_location(contactLoc)


import imageio

def create_video_from_rgba(images, output_path, fps=5):
    """
    Create an MP4 video from an RGBA image array.

    Args:
        images (list): List of RGBA image arrays.
        output_path (str): Path to save the resulting MP4 video.
        fps (int): Frames per second for the video (default: 200).
    """
    writer = imageio.get_writer(output_path, format='ffmpeg', fps=fps)

    for img in images:
        writer.append_data(img)

    writer.close()


image_array_list = []


import time
# visualize DDP warm-start
for t in range(N_ocp):
    # time.sleep(dt)
    viz.display(q_sol[t])

    for i in range(len(supportFeePos)):
        arrows[i].anchor_as_vector(supportFeePos[i], forces[i][t])
    

    image_array_list.append(viz.captureImage())



output_path = 'output.mp4'
create_video_from_rgba(image_array_list, output_path)