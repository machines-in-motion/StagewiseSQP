import pinocchio as pin
import numpy as np
import meshcat
import crocoddyl



class ResidualForce3D(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, nu):
        crocoddyl.ResidualModelAbstract.__init__(self, state, 12, nu, True, True, True)

    def calc(self, data, x, u=None):  
        data.r[:3] = data.shared.contacts.contacts['FL_FOOT_contact'].f.vector[:3]   
        data.r[3:6] = data.shared.contacts.contacts['FR_FOOT_contact'].f.vector[:3]   
        data.r[6:9] = data.shared.contacts.contacts['HL_FOOT_contact'].f.vector[:3]   
        data.r[9:12] = data.shared.contacts.contacts['HR_FOOT_contact'].f.vector[:3]   
        # # data.r = data.shared.differential.pinocchio.lambda_c 

    def calcDiff(self, data, x, u=None):
        # data.Rx = data.shared.differential.df_dx
        # data.Ru = data.shared.differential.df_du
        data.Rx[:3] = data.shared.contacts.contacts['FL_FOOT_contact'].df_dx[:3]   
        data.Rx[3:6] = data.shared.contacts.contacts['FR_FOOT_contact'].df_dx[:3]   
        data.Rx[6:9] = data.shared.contacts.contacts['HL_FOOT_contact'].df_dx[:3]   
        data.Rx[9:12] = data.shared.contacts.contacts['HR_FOOT_contact'].df_dx[:3]   
        
        data.Ru[:3] = data.shared.contacts.contacts['FL_FOOT_contact'].df_du[:3]   
        data.Ru[3:6] = data.shared.contacts.contacts['FR_FOOT_contact'].df_du[:3]   
        data.Ru[6:9] = data.shared.contacts.contacts['HL_FOOT_contact'].df_du[:3]   
        data.Ru[9:12] = data.shared.contacts.contacts['HR_FOOT_contact'].df_du[:3] 

class ResidualFrictionCone(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, mu, nu):
        crocoddyl.ResidualModelAbstract.__init__(self, state, 4, nu, True, True, True)
        self.mu = mu
        self.dcone_df = np.zeros((4, 12))

    def calc(self, data, x, u=None): 
        # constraint residual (expressed in constraint ref frame already)
        F = np.zeros(12) #data.shared.differential.pinocchio.lambda_c
        F[:3] = data.shared.contacts.contacts['FL_FOOT_contact'].f.vector[:3]   
        F[3:6] = data.shared.contacts.contacts['FR_FOOT_contact'].f.vector[:3]   
        F[6:9] = data.shared.contacts.contacts['HL_FOOT_contact'].f.vector[:3]   
        F[9:12] = data.shared.contacts.contacts['HR_FOOT_contact'].f.vector[:3]  

        data.r[0] = np.array([self.mu * F[2] - np.sqrt(F[0]**2 + F[1]**2)])
        data.r[1] = np.array([self.mu * F[5] - np.sqrt(F[3]**2 + F[4]**2)])
        data.r[2] = np.array([self.mu * F[8] - np.sqrt(F[6]**2 + F[7]**2)])
        data.r[3] = np.array([self.mu * F[11] - np.sqrt(F[9]**2 + F[10]**2)])

    def calcDiff(self, data, x, u=None):
        F = np.zeros(12) #data.shared.differential.pinocchio.lambda_c
        F[:3] = data.shared.contacts.contacts['FL_FOOT_contact'].f.vector[:3]   
        F[3:6] = data.shared.contacts.contacts['FR_FOOT_contact'].f.vector[:3]   
        F[6:9] = data.shared.contacts.contacts['HL_FOOT_contact'].f.vector[:3]   
        F[9:12] = data.shared.contacts.contacts['HR_FOOT_contact'].f.vector[:3]  

        self.dcone_df[0, 0] = -F[0] / np.sqrt(F[0]**2 + F[1]**2)
        self.dcone_df[0, 1] = -F[1] / np.sqrt(F[0]**2 + F[1]**2)
        self.dcone_df[0, 2] = self.mu

        self.dcone_df[1, 3] = -F[3] / np.sqrt(F[3]**2 + F[4]**2)
        self.dcone_df[1, 4] = -F[4] / np.sqrt(F[3]**2 + F[4]**2)
        self.dcone_df[1, 5] = self.mu

        self.dcone_df[2, 6] = -F[6] / np.sqrt(F[6]**2 + F[7]**2)
        self.dcone_df[2, 7] = -F[7] / np.sqrt(F[6]**2 + F[7]**2)
        self.dcone_df[2, 8] = self.mu

        self.dcone_df[3, 9] = -F[9] / np.sqrt(F[9]**2 + F[10]**2)
        self.dcone_df[3, 10] = -F[10] / np.sqrt(F[9]**2 + F[10]**2)
        self.dcone_df[3, 11] = self.mu

        df_dx = np.zeros((12, self.state.ndx))
        df_du = np.zeros((12, self.nu))
        df_dx[:3]  = data.shared.contacts.contacts['FL_FOOT_contact'].df_dx[:3]   
        df_dx[3:6] = data.shared.contacts.contacts['FR_FOOT_contact'].df_dx[:3]   
        df_dx[6:9] = data.shared.contacts.contacts['HL_FOOT_contact'].df_dx[:3]   
        df_dx[9:12] = data.shared.contacts.contacts['HR_FOOT_contact'].df_dx[:3]   
        
        df_du[:3] = data.shared.contacts.contacts['FL_FOOT_contact'].df_du[:3]   
        df_du[3:6] = data.shared.contacts.contacts['FR_FOOT_contact'].df_du[:3]   
        df_du[6:9] = data.shared.contacts.contacts['HL_FOOT_contact'].df_du[:3]   
        df_du[9:12] = data.shared.contacts.contacts['HR_FOOT_contact'].df_du[:3] 

        data.Rx = self.dcone_df @ df_dx 
        data.Ru = self.dcone_df @ df_du


def meshcat_material(r, g, b, a):
        material = meshcat.geometry.MeshPhongMaterial()
        material.color = int(r * 255) * 256 ** 2 + int(g * 255) * 256 + int(b * 255)
        material.opacity = a
        material.linewidth = 5.0
        return material

def addViewerBox(viz, name, sizex, sizey, sizez, rgba):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_object(meshcat.geometry.Box([sizex, sizey, sizez]),
                                meshcat_material(*rgba))


def addLineSegment(viz, name, vertices, rgba):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_object(meshcat.geometry.LineSegments(
                    meshcat.geometry.PointsGeometry(np.array(vertices)),     
                    meshcat_material(*rgba)
                    ))

def addPoint(viz, name, vertices, rgba):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_object(meshcat.geometry.Points(
                    meshcat.geometry.PointsGeometry(np.array(vertices)),     
                    meshcat_material(*rgba)
                    ))

def meshcat_transform(x, y, z, q, u, a, t):
    return np.array(pin.XYZQUATToSE3([x, y, z, q, u, a, t]))

def applyViewerConfiguration(viz, name, xyzquat):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_transform(meshcat_transform(*xyzquat))

def get_solution_trajectories(solver, rmodel, rdata, supportFeetIds, pinRefFrame=pin.LOCAL):
    xs, us = solver.xs, solver.us
    nq, nv, N = rmodel.nq, rmodel.nv, len(xs) 
    jointPos_sol = []
    jointVel_sol = []
    jointAcc_sol = []
    jointTorques_sol = []
    centroidal_sol = []

    x = []
    for time_idx in range (N):
        q, v = xs[time_idx][:nq], xs[time_idx][nq:]
        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.computeCentroidalMomentum(rmodel, rdata, q, v)
        centroidal_sol += [
            np.concatenate(
                [pin.centerOfMass(rmodel, rdata, q, v), np.array(rdata.hg)]
                )
                ]
        jointPos_sol += [q]
        jointVel_sol += [v]
        x += [xs[time_idx]]
        if time_idx < N-1:
            jointAcc_sol +=  [solver.problem.runningDatas[time_idx].xnext[nq::]] 
            jointTorques_sol += [us[time_idx]]




    sol = {'x':x, 'centroidal':centroidal_sol, 'jointPos':jointPos_sol, 
                      'jointVel':jointVel_sol, 'jointAcc':jointAcc_sol, 
                        'jointTorques':jointTorques_sol}        
    

    for frame_idx in supportFeetIds:
        # print('extract foot id ', frame_idx, "_name = ", rmodel.frames[frame_idx].name)
        ct_frame_name = rmodel.frames[frame_idx].name + "_contact"
        datas = [solver.problem.runningDatas[i].differential.multibody.contacts.contacts[ct_frame_name] for i in range(N-1)]
        ee_forces = [datas[k].jMf.actInv(datas[k].f).vector for k in range(N-1)] 
        sol[ct_frame_name] = [ee_forces[i] for i in range(N-1)]     
    
    return sol    


import numpy as np
import meshcat.geometry as g
import meshcat.transformations as tf

class Arrow(object):
    def __init__(self, meshcat_vis, name, 
                 location=[0,0,0], 
                 vector=[0,0,1],
                 length_scale=1,
                 color=0xff0000):

        self.vis = meshcat_vis[name]
        self.cone = self.vis["cone"]
        self.line = self.vis["line"]
        self.material = g.MeshBasicMaterial(color=color, reflectivity=0.5)
        
        self.location, self.length_scale = location, length_scale
        self.anchor_as_vector(location, vector)
    
    def _update(self):
        # pass
        translation = tf.translation_matrix(self.location)
        rotation = self.orientation
        offset = tf.translation_matrix([0, self.length/2, 0])
        self.pose = translation @ rotation @ offset
        self.vis.set_transform(self.pose)
        
    def set_length(self, length, update=True):
        self.length = length * self.length_scale
        cone_scale = self.length/0.08
        self.line.set_object(g.Cylinder(height=self.length, radius=0.005), self.material)
        self.cone.set_object(g.Cylinder(height=0.015, 
                                        radius=0.01, 
                                        radiusTop=0., 
                                        radiusBottom=0.01),
                             self.material)
        self.cone.set_transform(tf.translation_matrix([0.,cone_scale*0.04,0]))
        if update:
            self._update()
        
    def set_direction(self, direction, update=True):
        orientation = np.eye(4)
        orientation[:3, 0] = np.cross([1,0,0], direction)
        orientation[:3, 1] = direction
        orientation[:3, 2] = np.cross(orientation[:3, 0], orientation[:3, 1])
        self.orientation = orientation
        if update:
            self._update()
    
    def set_location(self, location, update=True):
        self.location = location
        if update:
            self._update()
        
    def anchor_as_vector(self, location, vector, update=True):
        self.set_direction(np.array(vector)/np.linalg.norm(vector), False)
        self.set_location(location, False)
        self.set_length(np.linalg.norm(vector), False)
        if update:
            self._update()

    def delete(self):
        self.vis.delete()




class Cone(object):
    def __init__(self, meshcat_vis, name,
                 location=[0,0,0], mu=1,
                 vector=[0,0,1],
                 length_scale=0.06):
        
        self.vis = meshcat_vis[name]
        self.cone = self.vis["cone"]
        self.material = g.MeshBasicMaterial(color=0xffffff, opacity = 0.5, reflectivity=0.5)


        self.mu = mu * length_scale
        self.location, self.length_scale = location, length_scale
        self.anchor_as_vector(location, vector)
    
    def _update(self):
        # pass
        translation = tf.translation_matrix(self.location)
        rotation = self.orientation
        offset = tf.translation_matrix([0, self.length/2, 0])
        self.pose = translation @ rotation @ offset
        self.vis.set_transform(self.pose)
        
    def set_length(self, length, update=True):
        self.length = length * self.length_scale
        cone_scale = self.length
        self.cone.set_object(g.Cylinder(height=cone_scale, 
                                        radius=self.mu, 
                                        radiusTop=self.mu, 
                                        radiusBottom=0),
                             self.material)
        # self.cone.set_transform(tf.translation_matrix([0.,cone_scale*0.04,0]))
        if update:
            self._update()
        
    def set_direction(self, direction, update=True):
        orientation = np.eye(4)
        orientation[:3, 0] = np.cross([1,0,0], direction)
        orientation[:3, 1] = direction
        orientation[:3, 2] = np.cross(orientation[:3, 0], orientation[:3, 1])
        self.orientation = orientation
        if update:
            self._update()
    
    def set_location(self, location, update=True):
        self.location = location
        if update:
            self._update()
        
    def anchor_as_vector(self, location, vector, update=True):
        self.set_direction(np.array(vector)/np.linalg.norm(vector), False)
        self.set_location(location, False)
        self.set_length(np.linalg.norm(vector), False)
        if update:
            self._update()

    def delete(self):
        self.vis.delete()