"""author: Ahmad Gazar"""
'''
copy-pasted from climbing_demo_ahmad
'''
import numpy as np
import crocoddyl
import pinocchio 
import pinocchio as pin

import example_robot_data 
from robot_properties_solo.solo12wrapper import Solo12Config



class Debris():
    def __init__(
            self, CONTACT, t_start=0.0, t_end=1.0,  x=None, y=None, z=None, axis=None, angle=None, ACTIVE=False
            ):
        if ACTIVE:
            STEP = 1.0
            axis = np.array(axis, np.float64)
            axis /= np.linalg.norm(axis)
            self.axis = axis
            self.pose = pin.SE3(pin.AngleAxis(angle, np.concatenate([axis, [0]])).matrix(),
                            np.array([x * STEP, y * STEP, z]))
        self.t_start = t_start 
        self.t_end = t_end
        self.CONTACT = CONTACT
        self.ACTIVE = ACTIVE 
        self.__fill_contact_idx()

    def __fill_contact_idx(self):
        if self.CONTACT == 'RF' or self.CONTACT == 'FR':
            self.idx = 0
        elif self.CONTACT == 'LF' or self.CONTACT == 'FL':
            self.idx = 1
        elif self.CONTACT == 'HR':
            self.idx = 2
        elif self.CONTACT == 'HL':
            self.idx = 3                                     
    
# given a contact plan, fill a contact trajectory    
def create_contact_trajectory(conf):
    contact_sequence = conf.contact_sequence
    contact_trajectory = dict([(foot.CONTACT, []) for foot in  contact_sequence[0]])
    for contacts in contact_sequence:
        for contact in contacts:
            contact_duration = int(round((contact.t_end-contact.t_start)/conf.dt))  
            for _ in range(contact_duration):
                contact_trajectory[contact.CONTACT].append(contact)  
    return contact_trajectory                


def create_contact_sequence(dt, gait, ee_frame_names, rmodel, rdata, q0):
    gait_templates = []
    steps = gait['nbSteps']
    if gait['type'] == 'TROT':
      for step in range (steps):
          if step < steps-1:
                gait_templates += [
                    ['doubleSupport', 'rflhStep', 'doubleSupport', 'lfrhStep']
                    ]
          else:
                gait_templates += [
                    ['doubleSupport', 'rflhStep', 'doubleSupport', 'lfrhStep', 'doubleSupport']
                    ]
    elif gait['type'] =='PACE':
        if rmodel.type == 'QUADRUPDED':
            for step in range (steps):
                if step < steps-1:
                    gait_templates += [
                        ['doubleSupport', 'rfrhStep', 'doubleSupport', 'lflhStep']
                        ]
                else:
                    gait_templates += [
                        ['doubleSupport','rfrhStep', 'doubleSupport', 'lflhStep', 'doubleSupport']
                        ]
        elif rmodel.type == 'HUMANOID':
            for step in range (steps):
                if step < steps-1:
                    gait_templates += [
                        ['doubleSupport', 'rfStep', 'doubleSupport', 'lfStep']
                        ]
                else:
                    gait_templates += [
                        ['doubleSupport','rfStep', 'doubleSupport', 'lfStep', 'doubleSupport']
                        ]                           
    elif gait['type'] == 'BOUND':
        for step in range (steps):
            if step < steps-1:
                gait_templates += [
                    ['doubleSupport', 'rflfStep', 'doubleSupport', 'rhlhStep']
                    ]
            else:
                gait_templates += [
                    ['doubleSupport', 'rflfStep', 'doubleSupport', 'rhlhStep', 'doubleSupport']
                    ]              
    elif gait['type'] == 'JUMP':
        for step in range (steps):
            gait_templates += [
              ['doubleSupport', 'NONE', 'doubleSupport']
              ]                  
    pin.forwardKinematics(rmodel, rdata, q0)
    pin.updateFramePlacements(rmodel, rdata)
    if rmodel.type == 'QUADRUPED':
      hlFootPos = rdata.oMf[rmodel.getFrameId(ee_frame_names[2])].translation
      hrFootPos = rdata.oMf[rmodel.getFrameId(ee_frame_names[3])].translation
    flFootPos = rdata.oMf[rmodel.getFrameId(ee_frame_names[0])].translation
    frFootPos = rdata.oMf[rmodel.getFrameId(ee_frame_names[1])].translation
    t_start = 0.0 
    contact_sequence = []
    stepKnots, supportKnots = gait['stepKnots'], gait['supportKnots']
    stepLength = gait['stepLength']
    for gait in gait_templates:
        for phase in gait:
            contact_sequence_k = []
            if rmodel.type == 'QUADRUPED' and phase == 'doubleSupport':
              t_end = t_start + supportKnots*dt
              contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, x=frFootPos[0], 
                                    y=frFootPos[1], z=frFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
              contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, x=flFootPos[0], 
                                    y=flFootPos[1], z=flFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
              contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, x=hrFootPos[0],
                                    y=hrFootPos[1], z=hrFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
              contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, x=hlFootPos[0],  
                                    y=hlFootPos[1], z=hlFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
            elif rmodel.type == 'HUMANOID'  and phase == 'doubleSupport':
              t_end = t_start + supportKnots*dt
              contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, x=frFootPos[0], 
                                    y=frFootPos[1], z=frFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
              contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, x=flFootPos[0],  
                                    y=flFootPos[1], z=flFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))        
            elif phase == 'rflhStep':
              t_end = t_start + stepKnots*dt
              contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, ACTIVE=False))
              contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, x=flFootPos[0],  
                                    y=flFootPos[1], z=flFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
              contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, x=hrFootPos[0], 
                                    y=hrFootPos[1], z=hrFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
              contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, ACTIVE=False))
              frFootPos[0] += stepLength
              hlFootPos[0] += stepLength
            elif phase == 'rfStep':
              t_end = t_start + stepKnots*dt
              contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, ACTIVE=False))
              contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, x=flFootPos[0], 
                                    y=flFootPos[1], z=flFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
              frFootPos[0] += stepLength
            elif phase == 'lfrhStep':
              t_end = t_start + stepKnots*dt
              contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, x=frFootPos[0],
                                    y=frFootPos[1], z=frFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
              contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, ACTIVE=False))
              contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, ACTIVE=False))
              contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, x=hlFootPos[0], 
                                    y=hlFootPos[1], z=hlFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
              flFootPos[0] += stepLength
              hrFootPos[0] += stepLength      
            elif phase == 'lfStep':
              t_end = t_start + stepKnots*dt
              contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, x=frFootPos[0],  
                                    y=frFootPos[1], z=frFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
              contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, ACTIVE=False))
              flFootPos[0] += stepLength
            elif phase == 'rfrhStep':
              t_end = t_start + stepKnots*dt
              contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, ACTIVE=False))
              contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, x=flFootPos[0],  
                                    y=flFootPos[1], z=flFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
              contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, ACTIVE=False))
              contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, x=hlFootPos[0],  
                                    y=hlFootPos[1], z=hlFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
              frFootPos[0] += stepLength
              hrFootPos[0] += stepLength      
            elif phase == 'lflhStep':
              t_end = t_start + stepKnots*dt
              contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, x=frFootPos[0],  
                                    y=frFootPos[1], z=frFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
              contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, ACTIVE=False))
              contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, x=hrFootPos[0], 
                                    y=hrFootPos[1], z=hrFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
              contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, ACTIVE=False))
              flFootPos[0] += stepLength
              hlFootPos[0] += stepLength
            elif phase == 'rflfStep':
              t_end = t_start + stepKnots*dt
              contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, ACTIVE=False))
              contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, ACTIVE=False))
              contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, x=hrFootPos[0],  
                                    y=hrFootPos[1], z=hrFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
              contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, x=hlFootPos[0],  
                                    y=hlFootPos[1], z=hlFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
              frFootPos[0] += stepLength
              flFootPos[0] += stepLength      
            elif phase == 'rhlhStep':
              t_end = t_start + stepKnots*dt
              contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, x=frFootPos[0],  
                                    y=frFootPos[1], z=frFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
              contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, x=flFootPos[0], 
                                    y=flFootPos[1], z=flFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))                        
              contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, ACTIVE=False))
              contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, ACTIVE=False))
              hrFootPos[0] += stepLength
              hlFootPos[0] += stepLength
            else:
              t_end = t_start + stepKnots*dt
              contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, ACTIVE=False))
              contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, ACTIVE=False))
              frFootPos[0] += stepLength
              flFootPos[0] += stepLength
              if rmodel.type == 'QUADRUPED':
                  hrFootPos[0] += stepLength      
                  hlFootPos[0] += stepLength
                  contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, ACTIVE=False))
                  contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, ACTIVE=False)) 
            t_start = t_end
            contact_sequence += [contact_sequence_k] 
    return gait_templates, contact_sequence


def create_climbing_contact_sequence(dt, gait, ee_frame_names, rmodel, rdata, q0):
      gait_templates = []
      steps = gait['nbSteps']
      if gait['type'] == 'TROT':
            for step in range (steps):
                  if step < steps-1:
                        gait_templates += [['doubleSupport', 'rflhStep', 'doubleSupport', 'lfrhStep']]
                  else:
                        gait_templates += [['doubleSupport', 
                                            'rflhStep', 'doubleSupport', 
                                            'lfrhStep', 'doubleSupport']]          
      pin.forwardKinematics(rmodel, rdata, q0)
      pin.updateFramePlacements(rmodel, rdata)
      if rmodel.type == 'QUADRUPED':
        hlFootPos = rdata.oMf[rmodel.getFrameId(ee_frame_names[2])].translation
        hrFootPos = rdata.oMf[rmodel.getFrameId(ee_frame_names[3])].translation
      flFootPos = rdata.oMf[rmodel.getFrameId(ee_frame_names[0])].translation
      frFootPos = rdata.oMf[rmodel.getFrameId(ee_frame_names[1])].translation
      t_start = 0.0 
      contact_sequence = []
      stepKnots, supportKnots = gait['stepKnots'], gait['supportKnots']
      stepLength = gait['stepLength']
      stepHeight = gait['stepHeight']
      for gait in gait_templates:
            for phase in gait:
                  contact_sequence_k = []
                  if rmodel.type == 'QUADRUPED' and phase == 'doubleSupport':
                    t_end = t_start + supportKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, x=frFootPos[0], 
                                         y=frFootPos[1], z=frFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, x=flFootPos[0], 
                                         y=flFootPos[1], z=flFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, x=hrFootPos[0],
                                          y=hrFootPos[1], z=hrFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, x=hlFootPos[0],  
                                          y=hlFootPos[1], z=hlFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                  elif rmodel.type == 'HUMANOID'  and phase == 'doubleSupport':
                    t_end = t_start + supportKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, x=frFootPos[0], 
                                          y=frFootPos[1], z=frFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, x=flFootPos[0],  
                                          y=flFootPos[1], z=flFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))        
                  elif phase == 'rflhStep':
                    t_end = t_start + stepKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, x=flFootPos[0],  
                                          y=flFootPos[1], z=flFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, x=hrFootPos[0], 
                                          y=hrFootPos[1], z=hrFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, ACTIVE=False))
                    frFootPos[0] += stepLength
                    hlFootPos[0] += stepLength
                    frFootPos[2] += stepHeight
                    hlFootPos[2] += stepHeight
                  elif phase == 'rfStep':
                    t_end = t_start + stepKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, x=flFootPos[0], 
                                          y=flFootPos[1], z=flFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    frFootPos[0] += stepLength
                    frFootPos[2] += stepHeight
                  elif phase == 'lfrhStep':
                    t_end = t_start + stepKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, x=frFootPos[0],
                                          y=frFootPos[1], z=frFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, x=hlFootPos[0], 
                                          y=hlFootPos[1], z=hlFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    flFootPos[0] += stepLength
                    hrFootPos[0] += stepLength
                    flFootPos[2] += stepHeight
                    hrFootPos[2] += stepHeight          
                  elif phase == 'lfStep':
                    t_end = t_start + stepKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, x=frFootPos[0],  
                                          y=frFootPos[1], z=frFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, ACTIVE=False))
                    flFootPos[0] += stepLength
                    flFootPos[2] += stepHeight
                  elif phase == 'rfrhStep':
                    t_end = t_start + stepKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, x=flFootPos[0],  
                                         y=flFootPos[1], z=flFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, x=hlFootPos[0],  
                                         y=hlFootPos[1], z=hlFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    frFootPos[0] += stepLength
                    hrFootPos[0] += stepLength
                    frFootPos[2] += stepHeight
                    hrFootPos[2] += stepHeight        
                  elif phase == 'lflhStep':
                    t_end = t_start + stepKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, x=frFootPos[0],  
                                          y=frFootPos[1], z=frFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, x=hrFootPos[0], 
                                          y=hrFootPos[1], z=hrFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, ACTIVE=False))
                    flFootPos[0] += stepLength
                    hlFootPos[0] += stepLength
                    flFootPos[2] += stepHeight
                    hlFootPos[2] += stepHeight
                  elif phase == 'rflfStep':
                    t_end = t_start + stepKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, x=hrFootPos[0],  
                                          y=hrFootPos[1], z=hrFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, x=hlFootPos[0],  
                                          y=hlFootPos[1], z=hlFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    frFootPos[0] += stepLength
                    flFootPos[0] += stepLength
                    frFootPos[2] += stepHeight
                    flFootPos[2] += stepHeight      
                  elif phase == 'rhlhStep':
                    t_end = t_start + stepKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, x=frFootPos[0],  
                                          y=frFootPos[1], z=frFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, x=flFootPos[0], 
                                          y=flFootPos[1], z=flFootPos[2], axis=[-1, 0], angle=0.0, ACTIVE=True))                        
                    contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, ACTIVE=False))
                    hrFootPos[0] += stepLength
                    hlFootPos[0] += stepLength
                    hrFootPos[2] += stepHeight
                    hlFootPos[2] += stepHeight
                  else:
                    t_end = t_start + stepKnots*dt
                    contact_sequence_k.append(Debris(CONTACT='FR', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='FL', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='HR', t_start=t_start, t_end=t_end, ACTIVE=False))
                    contact_sequence_k.append(Debris(CONTACT='HL', t_start=t_start, t_end=t_end, ACTIVE=False))
                    frFootPos[0] += stepLength
                    flFootPos[0] += stepLength
                    frFootPos[2] += stepHeight
                    flFootPos[2] += stepHeight
                    if rmodel.type == 'QUADRUPED':
                        hrFootPos[0] += stepLength      
                        hlFootPos[0] += stepLength
                        hrFootPos[2] += stepHeight      
                        hlFootPos[2] += stepHeight  
                  t_start = t_end
                  contact_sequence += [contact_sequence_k] 
      return gait_templates, contact_sequence


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



class WholeBodyDDPSolver:
    # constructor
    def __init__(self, model, x0, centroidalTask=None, forceTask=None, MPC=False, WARM_START=True):
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
        self.x0 = x0 #model.x0
        # extra tasks
        self.centroidal_task = centroidalTask
        self.force_task = forceTask
        # flags
        self.WARM_START = WARM_START
        self.RECEEDING_HORIZON = MPC
        # initialize ocp and create DDP solver
        self.__add_tracking_tasks(centroidalTask, forceTask)
        self.pb = self.__init_ocp_and_solver()
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
            return ocp
            # self.solver = crocoddyl.SolverFDDP(ocp)#GNMSCPP(ocp, VERBOSE=True)
            # # self.solver.with_callbacks = True
            # self.solver.termination_tol = 1e-4
            # self.solver.setCallbacks([crocoddyl.CallbackLogger(),
            #                         crocoddyl.CallbackVerbose()])     
    
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
    
    def solve(self, solver, x_warm_start=False, u_warm_start=False, max_iter=100):
        # solver = self.solver
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
