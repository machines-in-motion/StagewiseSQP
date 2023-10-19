import crocoddyl
import pinocchio
import numpy as np
import example_robot_data 
from robot_properties_solo.solo12wrapper import Solo12Config
import pinocchio as pin
import standing_utils
# import sobec

import sys
import mim_solvers

pinRef        = pin.LOCAL_WORLD_ALIGNED
FORCE_CSTR    = False
FRICTION_CSTR = False
MU = 0.8
PLOT = False
PLAY = False
SAVE = True

SOLVE_OCP = False


PLOT_1 = False
PLOT_2 = False 
PLOT_3 = False
PLOT_4 = True

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

N_ocp = 250 #100
dt = 0.02
T = N_ocp * dt
radius = 0.065
for t in range(N_ocp+1):
    comDes_t = comRef.copy()
    w = (2 * np.pi) * 0.2 # / T
    comDes_t[0] += radius * np.sin(w * t * dt) 
    comDes_t[1] += radius * (np.cos(w * t * dt) - 1)
    comDes += [comDes_t]

running_models = []
constraintModels = []
for t in range(N_ocp+1):
    contactModel = crocoddyl.ContactModelMultiple(state, nu)
    costModel = crocoddyl.CostModelSum(state, nu)

    # Add contact
    for frame_idx in supportFeetIds:
        support_contact = crocoddyl.ContactModel3D(state, frame_idx, np.array([0., 0., 0.]), pinRef, nu, np.array([0., 0.]))
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

    # Add contact force constraint >= 0 & friction cone 
    n_cstr = 0
    constraintModelManager = crocoddyl.ConstraintModelManager(state, actuation.nu)
    if(t != N_ocp):
        if(FORCE_CSTR):
            clip_force_min = np.array([-np.inf, -np.inf, -np.inf]*4)
            clip_force_max = np.array([np.inf, np.inf, np.inf]*4)
            residualForce = standing_utils.ResidualForce3D(state, actuation.nu)
            constraintForce = crocoddyl.ConstraintModelResidual(state, residualForce, clip_force_min, clip_force_max)
            constraintModelManager.addConstraint("force", constraintForce)
            n_cstr += 12
        if(FRICTION_CSTR):
            residualFriction = standing_utils.ResidualFrictionCone(state, MU, actuation.nu)
            constraintFriction = crocoddyl.ConstraintModelResidual(state, residualFriction, np.array([0.]*4), np.array([np.inf]*4))
            constraintModelManager.addConstraint("friction", constraintFriction)
            n_cstr += 4

    dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, costModel, constraintModelManager, 0., True)
    model = crocoddyl.IntegratedActionModelEuler(dmodel, dt)

    running_models += [model]

# Create shooting problem
ocp = crocoddyl.ShootingProblem(x0, running_models[:-1], running_models[-1])
fwoeihfe
# Create solver , warm-start and solve
if(FRICTION_CSTR):
    solver = mim_solvers.SolverCSQP(ocp)
    solver.max_qp_iters = 1000
    max_iter = 500
    solver.with_callbacks = True
    solver.use_filter_ls = True
    solver.filter_size = max_iter
    solver.termination_tolerance = 1e-4
    solver.eps_abs = 1e-6
    solver.eps_rel = 1e-6
else:
    solver = mim_solvers.SolverSQP(ocp)
    max_iter = 500
    solver.termination_tol = 1e-4
    solver.with_callbacks = True
    solver.use_filter_ls = True
    solver.filter_size = max_iter
solver.KKT = True

if(SOLVE_OCP):   
    xs = [x0]*(solver.problem.T + 1)
    us = solver.problem.quasiStatic([x0]*solver.problem.T) #[np.zeros(actuation.nu)]*solver.problem.T #solver.us #solver.problem.quasiStatic([x0]*solver.problem.T)
    solver.solve(xs, us, max_iter)   
    solution = standing_utils.get_solution_trajectories(solver, rmodel, rdata, supportFeetIds, pinRefFrame=pinRef)
    q_sol = solution['jointPos']
    centroidal_sol = solution['centroidal']
    import pickle 
    if(FRICTION_CSTR):
        name = '/tmp/sol_constrained_mu='+str(MU)+'.pkl'
    else:
        name = '/tmp/sol_unconstrained_mu='+str(MU)+'.pkl'
    with open(name, 'wb') as f:
        pickle.dump(solution, f)

    # Plot results
    if(PLOT):
        import matplotlib.pyplot as plt
        # Plot forces 
        time_lin = np.linspace(0, T, solver.problem.T)
        fig, axs = plt.subplots(4, 3, constrained_layout=True)
        for i, frame_idx in enumerate(supportFeetIds):
            ct_frame_name = rmodel.frames[frame_idx].name + "_contact"
            forces = np.array(solution[ct_frame_name])
            axs[i, 0].plot(time_lin, forces[:, 0], label="Fx")
            axs[i, 1].plot(time_lin, forces[:, 1], label="Fy")
            axs[i, 2].plot(time_lin, forces[:, 2], label="Fz")
            # Add friction cone constraints 
            Fz_lb = (1./MU)*np.sqrt(forces[:, 0]**2 + forces[:, 1]**2)
            # Fz_ub = np.zeros(time_lin.shape)
            # axs[i, 2].plot(time_lin, Fz_ub, 'k-.', label='ub')
            axs[i, 2].plot(time_lin, Fz_lb, 'k-.', label='lb')
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


        comDes = np.array(comDes)
        centroidal_sol = np.array(centroidal_sol)
        plt.figure()
        plt.plot(comDes[:, 0], comDes[:, 1], "--", label="reference")
        plt.plot(centroidal_sol[:, 0], centroidal_sol[:, 1], label="solution")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("COM trajectory")

        # fig, axs = plt.subplots(actuation.nu ,1)
        # torques = np.array(solution["jointTorques"])
        # for i in range(actuation.nu):
        #     axs[i].plot(time_lin, torques[:, i])
        #     axs[i].set_ylabel("$\\tau_{%s}$"%i)
        #     axs[i].grid()
        # fig.suptitle('Control input', fontsize=16)



        # time_lin = np.linspace(0, T, solver.problem.T+1)

        # jointPos = np.array(solution["jointPos"])
        # fig, axs = plt.subplots(rmodel.nq ,1)
        # for i in range(rmodel.nq):
        #     axs[i].plot(time_lin, jointPos[:, i])
        #     axs[i].set_ylabel("$q_{%s}$"%i)
        #     axs[i].grid()
        # fig.suptitle('Joint position', fontsize=16)

        # fig, axs = plt.subplots(3, 1, constrained_layout=True)
        # for i in range(3):
        #     axs[i].plot(time_lin, comDes[:, i], "--", label="reference")
        #     axs[i].plot(time_lin, centroidal_sol[:, i], label="solution")
        #     axs[i].grid()
        # axs[0].set_ylabel("$COM_{x}$")
        # axs[1].set_ylabel("$COM_{y}$")
        # axs[2].set_ylabel("$COM_{z}$")
        # plt.legend()    

        plt.title("COM trajectory")

        plt.show()

    if(PLAY):
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


        angle = 0.0  # Initial angle
        rotation_speed = 0.05  # Speed of rotation (adjust as needed)

        cam_pose = tf.translation_matrix([0, 0, 0.])  # Example camera position
        cam_pose[:3, :3] = tf.euler_matrix(0.0, 0.0, np.pi/3)[:3, :3]  # Example camera orientation
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

        cone1 = standing_utils.Cone(viz.viewer, "friction_cone_1", location=supportFeePos[0], mu=MU)
        cone2 = standing_utils.Cone(viz.viewer, "friction_cone_2", location=supportFeePos[1], mu=MU)
        cone3 = standing_utils.Cone(viz.viewer, "friction_cone_3", location=supportFeePos[2], mu=MU)
        cone4 = standing_utils.Cone(viz.viewer, "friction_cone_4", location=supportFeePos[3], mu=MU)

        arrows = [arrow1, arrow2, arrow3, arrow4]
        forces = []

        for i, contactLoc in enumerate(supportFeePos):
            ct_frame_name = rmodel.frames[supportFeetIds[i]].name + "_contact"
            forces.append(np.array(solution[ct_frame_name])[:, :3])
            arrows[i].set_location(contactLoc)


        image_array_list = []


        import time
        # visualize DDP warm-start
        time.sleep(5)
        for t in range(N_ocp):
            # time.sleep(dt)
            viz.display(q_sol[t])

            for i in range(len(supportFeePos)):
                arrows[i].anchor_as_vector(supportFeePos[i], forces[i][t])
            

            image_array_list.append(viz.captureImage())

        if(SAVE):
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

            output_path = 'output.mp4'
            create_video_from_rgba(image_array_list, output_path)

else:
    import matplotlib.pyplot as plt
    import pickle 
    with open('/tmp/sol_constrained_mu='+str(MU)+'.pkl', 'rb') as f:
        constrained_sol = pickle.load(f)
    with open('/tmp/sol_unconstrained_mu='+str(MU)+'.pkl', 'rb') as f:
        unconstrained_sol = pickle.load(f)

    # Plot forces Fx,Fy,Fz
    if(PLOT_1):
        time_lin = np.linspace(0, T, solver.problem.T)
        fig, axs = plt.subplots(4, 3, figsize=(19.2,10.8), constrained_layout=True)
        for i, frame_idx in enumerate(supportFeetIds):
            ct_frame_name = rmodel.frames[frame_idx].name + "_contact"
            forces1 = np.array(unconstrained_sol[ct_frame_name])
            forces2 = np.array(constrained_sol[ct_frame_name])
            # Plot unconstrained forces
            axs[i, 0].plot(time_lin, forces1[:, 0], color='g', linewidth=4,  alpha=0.5) 
            axs[i, 1].plot(time_lin, forces1[:, 1], color='g', linewidth=4,  alpha=0.5) 
            axs[i, 2].plot(time_lin, forces1[:, 2], color='g', linewidth=4, label='Unconstrained', alpha=0.5) 
            # Plot constrained forces
            axs[i, 0].plot(time_lin, forces2[:, 0], color='b', linewidth=4,  alpha=0.5) 
            axs[i, 1].plot(time_lin, forces2[:, 1], color='b', linewidth=4,  alpha=0.5) 
            axs[i, 2].plot(time_lin, forces2[:, 2], color='b', linewidth=4, label="Constrained", alpha=0.5) 
            # Add friction cone constraints 
            Fz_lb1 = (1./MU)*np.sqrt(forces1[:, 0]**2 + forces1[:, 1]**2)
            Fz_lb2 = (1./MU)*np.sqrt(forces2[:, 0]**2 + forces2[:, 1]**2)
            axs[i, 2].plot(time_lin, Fz_lb1, color='k', linestyle='--', linewidth=4, label='Friction cone (unconstrained)', alpha=0.5)
            axs[i, 2].plot(time_lin, Fz_lb2, color='r', linestyle='--', linewidth=4, label='Friction cone (constrained)', alpha=0.5)
            

            axs[i, 0].tick_params(axis = 'y', labelsize=18)
            axs[i, 1].tick_params(axis = 'y', labelsize=18)
            axs[i, 2].tick_params(axis = 'y', labelsize=18)
            if(i != 3):
                axs[i,0].xaxis.set_tick_params(labelbottom=False)
                axs[i,1].xaxis.set_tick_params(labelbottom=False)
                axs[i,2].xaxis.set_tick_params(labelbottom=False)
            axs[i, 0].grid()
            axs[i, 1].grid()
            axs[i, 2].grid()

            axs[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
            axs[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
            axs[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
            axs[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
            axs[i,2].yaxis.set_major_locator(plt.MaxNLocator(2))
            axs[i,2].yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))

        axs[0, 0].set_ylabel('FL', fontsize=22)
        axs[1, 0].set_ylabel('FR', fontsize=22)
        axs[2, 0].set_ylabel('HL', fontsize=22)
        axs[3, 0].set_ylabel('HR', fontsize=22)

        handles, labels = axs[-1, 2].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.36, 0.4), prop={'size': 22}) 
        # axs[-1, 2].legend(loc='upper left', bbox_to_anchor=(-1., 0.885), prop={'size': 22})

        fig.align_ylabels(axs[:,0])
        fig.align_ylabels(axs[:,1])
        fig.align_ylabels(axs[:,2])
        
        axs[-1, 0].tick_params(axis = 'x', labelsize=18)
        axs[-1, 1].tick_params(axis = 'x', labelsize=18)
        axs[-1, 2].tick_params(axis = 'x', labelsize=18)

        axs[0, 0].text(2., 1.2, r"$F_x$", fontdict={'size':26})
        axs[0, 1].text(2., 2.5, r"$F_y$", fontdict={'size':26})
        axs[0, 2].text(2., 8, r"$F_z$", fontdict={'size':26})


        # axs[3, 0].set_xlabel('Time (s)', fontsize=22)
        axs[3, 1].set_xlabel('Time (s)', fontsize=22)
        # axs[3, 2].set_xlabel('Time (s)', fontsize=22)
        # fig.savefig('/home/skleff/data_paper_CSSQP/solo_standing_friction.pdf', bbox_inches="tight")
        # fig.suptitle('Force', fontsize=16)

    if(PLOT_2):
        time_lin = np.linspace(0, T, solver.problem.T)
        fig, axs = plt.subplots(1, 2, figsize=(19.2,10.8), constrained_layout=True)
        for i, ct_frame_name in enumerate(['FL_FOOT_contact', 'HL_FOOT_contact']):
            # ct_frame_name = rmodel.frames[frame_idx].name + "_contact"
            forces1 = np.array(unconstrained_sol[ct_frame_name])
            forces2 = np.array(constrained_sol[ct_frame_name])
            # Plot unconstrained forces
            axs[i].plot(forces1[:, 0]/forces1[:, 2], forces1[:, 1]/forces1[:, 2], color='g', linewidth=4, label='Unconstrained', alpha=0.5) 
            axs[i].plot(forces2[:, 0]/forces2[:, 2], forces2[:, 1]/forces2[:, 2], color='b', linewidth=4, label='Constrained', alpha=0.5) 
            # Add friction cone constraints 
            theta = np.linspace(0, 2*np.pi, 100)
            x1 = MU*np.cos(theta)
            x2 = MU*np.sin(theta)
            axs[i].plot(x1, x2, color='k', linestyle='--', linewidth=4, label='Friction cone', alpha=0.5)
            axs[i].tick_params(axis = 'y', labelsize=18)
            axs[i].grid()
            axs[i].yaxis.set_major_locator(plt.MaxNLocator(2))
            axs[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
            axs[i].axis('equal')
            axs[i].yaxis.set_major_locator(plt.MaxNLocator(2))
            axs[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))

    # Current paper plot : FL and HL force ratio
    if(PLOT_3):
        time_lin = np.linspace(0, T, solver.problem.T)
        fig, axs = plt.subplots(1, 2, figsize=(19.2,10.8), constrained_layout=True)
        for i, ct_frame_name in enumerate(['FL_FOOT_contact', 'HL_FOOT_contact']):
            forces1 = np.array(unconstrained_sol[ct_frame_name])
            forces2 = np.array(constrained_sol[ct_frame_name])
            # Plot unconstrained forces
            f1 = np.sqrt( ( forces1[:, 0]**2 + forces1[:, 1]**2 ) / forces1[:, 2]**2 )
            f2 = np.sqrt( ( forces2[:, 0]**2 + forces2[:, 1]**2 ) / forces2[:, 2]**2 )
            axs[i].plot(time_lin, f1, color='g', linewidth=4, label='Unconstrained', alpha=0.5) 
            axs[i].plot(time_lin, f2, color='b', linewidth=4, label='Constrained', alpha=0.5) 
            # Add friction cone constraints 
            axs[i].plot(time_lin, [MU]*solver.problem.T, color='k', linestyle='--', linewidth=4, label='Friction cone ('+r"$\mu$"+'='+str(MU)+')', alpha=0.5)
            axs[i].tick_params(axis = 'y', labelsize=18)
            # if(i != 3):
                # axs[i%2, i//2].xaxis.set_tick_params(labelbottom=False)
            axs[i].grid()
            # axs[i%2, i//2].set_ylim(0., 5.5)
            # axs[i].set_ylim(0., 1)
            # axs[i].yaxis.set_major_locator(plt.MaxNLocator(2))
            # axs[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))

        axs[0].set_ylabel('FL force ratio', fontsize=22)
        axs[1].set_ylabel('HL force ratio', fontsize=22)

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left', prop={'size': 22}) 
        # axs[-1, 2].legend(loc='upper left', bbox_to_anchor=(-1., 0.885), prop={'size': 22})

        fig.align_ylabels(axs[:])
        fig.align_xlabels(axs[:])
        
        axs[0].tick_params(axis = 'x', labelsize=18)
        axs[1].tick_params(axis = 'x', labelsize=18)

        # axs[0].text(2., 6, 'HL', fontdict={'size':30})
        # axs[1].text(2., 6, 'FL', fontdict={'size':30})


        axs[0].set_xlabel('Time (s)', fontsize=22)
        axs[1].set_xlabel('Time (s)', fontsize=22)
        # fig.savefig('/home/skleff/data_paper_CSSQP/solo_standing_friction_normalized.pdf', bbox_inches="tight")

    # Only FL
    if(PLOT_4):
        time_lin = np.linspace(0, T, solver.problem.T)
        fig, axs = plt.subplots(1, 1, figsize=(19.2,10.8), constrained_layout=True)
        forces1 = np.array(unconstrained_sol['FL_FOOT_contact'])
        forces2 = np.array(constrained_sol['FL_FOOT_contact'])
        # Plot unconstrained forces
        f1 = np.sqrt( ( forces1[:, 0]**2 + forces1[:, 1]**2 ) / forces1[:, 2]**2 )
        f2 = np.sqrt( ( forces2[:, 0]**2 + forces2[:, 1]**2 ) / forces2[:, 2]**2 )
        axs.plot(time_lin, f1, color='g', linewidth=4, label='Unconstrained', alpha=0.5) 
        axs.plot(time_lin, f2, color='b', linewidth=4, label='Constrained', alpha=0.5) 
        # Add friction cone constraints 
        axs.plot(time_lin, [MU]*solver.problem.T, color='k', linestyle='--', linewidth=4, label='Friction cone ('+r"$\mu$"+'='+str(MU)+')', alpha=0.5)
        axs.grid()
        MAX = 5.3
        axs.axhspan(0.8, MAX, -MAX, MAX, color='gray', alpha=0.2, lw=0)
        axs.set_xlim(0., 5)
        axs.set_ylim(0., 5.3)
        axs.set_ylabel('FL force ratio', fontsize=22)
        handles, labels = axs.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.03, 1.), prop={'size': 26}) 
        axs.tick_params(axis = 'x', labelsize=22)
        axs.tick_params(axis = 'y', labelsize=22)
        axs.set_xlabel('Time (s)', fontsize=26)
        # fig.savefig('/home/skleff/data_paper_CSSQP/solo_standing_friction_normalized2.pdf', bbox_inches="tight")

    plt.show()
