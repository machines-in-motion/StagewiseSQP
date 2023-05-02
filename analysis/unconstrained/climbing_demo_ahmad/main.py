"""author: Ahmad Gazar"""

from robot_properties_solo.solo12wrapper import Solo12Config
from wholebody_croccodyl_solver import WholeBodyDDPSolver
from wholebody_croccodyl_model import WholeBodyModel
import pinocchio as pin
import numpy as np
import utils
import conf
import sys

# DDP warm-start
wbd_model = WholeBodyModel(conf)
ddp_planner = WholeBodyDDPSolver(wbd_model, MPC=False, WARM_START=False)
ddp_planner.solve(max_iter=400)
ddp_sol = ddp_planner.get_solution_trajectories()
q_warmstart = ddp_sol['jointPos']
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
# add contact surfaces
s = 0.5*conf.step_adjustment_bound
for i, contacts in enumerate(conf.contact_sequence):
    for contact_idx, contact in enumerate(contacts):
        if contact.ACTIVE:
            t = contact.pose.translation
            # debris box
            if contact.CONTACT == 'FR' or contact.CONTACT == 'FL':
                utils.addViewerBox(
                    viz, 'world/debris'+str(i)+str(contact_idx), 
                    2*s, 2*s, 0., [1., .2, .2, .5]
                    )
            if contact.CONTACT == 'HR' or contact.CONTACT == 'HL':
                utils.addViewerBox(
                    viz, 'world/debris'+str(i)+str(contact_idx),
                    2*s, 2*s, 0., [.2, .2, 1., .5]
                    )
            utils.applyViewerConfiguration(
                viz, 'world/debris'+str(i)+str(contact_idx), 
                [t[0], t[1], t[2]-0.017, 1, 0, 0, 0]
                )
            utils.applyViewerConfiguration(
                viz, 'world/debris_center'+str(i)+str(contact_idx), 
                [t[0], t[1], t[2]-0.017, 1, 0, 0, 0]
                ) 
# visualize DDP warm-start
for q_warmstart_k in q_warmstart:
    for i in range(int(conf.dt/conf.dt_ctrl)):
        viz.display(q_warmstart_k)