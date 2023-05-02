"""author: Ahmad Gazar"""

import numpy as np
import pinocchio as pin 
import example_robot_data 
from contact_plan import create_climbing_contact_sequence
from robot_properties_solo.solo12wrapper import Solo12Config

# walking parameters:
# -------------------
dt = 0.01
dt_ctrl = 0.001
gait ={'type': 'TROT',
      'stepLength' : 0.1, 
      'stepHeight' : 0.05,
      'stepKnots' : 15,
      'supportKnots' : 10,
      'nbSteps': 3}
mu = 0.5 # linear friction coefficient

# robot model and parameters
# --------------------------
robot_name = 'solo12'
ee_frame_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
solo12 = example_robot_data.ROBOTS[robot_name]()
rmodel = solo12.robot.model
rmodel.type = 'QUADRUPED'
rmodel.foot_type = 'POINT_FOOT'
rdata = rmodel.createData()
robot_mass = pin.computeTotalMass(rmodel)

gravity_constant = -9.81 
max_leg_length = 0.34
step_adjustment_bound = 0.07                         

q0 = np.array(Solo12Config.initial_configuration.copy())
q0[0] = 0.0
gait_templates, contact_sequence = create_climbing_contact_sequence(
      dt, gait, ee_frame_names, rmodel, rdata, q0
      )
# planning and control horizon lengths:   
# -------------------------------------
N = int(round(contact_sequence[-1][0].t_end/dt, 2))
N_mpc = (gait['stepKnots'] + (gait['supportKnots']))*3
N_mpc_wbd = int(round(N_mpc/2, 2))
N_ctrl = int((N-1)*(dt/dt_ctrl))    

# whole-body cost objective weights:
# ---------------------------------- 
freeFlyerQWeight = [0.]*3 + [500.]*3
freeFlyerVWeight = [10.]*6
legsQWeight = [0.01]*(rmodel.nv - 6)
legsWWeights = [1.]*(rmodel.nv - 6)
wbd_state_reg_weights = np.array(
      freeFlyerQWeight + legsQWeight + freeFlyerVWeight + legsWWeights
      )         

whole_body_task_weights = {
                            'swingFoot':{'preImpact':{'position':1e7,'velocity':0e1}, 
                                            'impact':{'position':1e7,'velocity':5e5}
                                           }, 
                            'comTrack':1e5, 'stateBounds':1e3, 'centroidalTrack': 1e4, 
                            'stateReg':{'stance':1e-1, 'impact':1e0}, 'ctrlReg':{'stance':1e-3, 'impact':1e-2}, 
                            'frictionCone':20, 'contactForceTrack':100
                            }                                                                        