"""author: Ahmad Gazar"""

import numpy as np
import pinocchio as pin

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