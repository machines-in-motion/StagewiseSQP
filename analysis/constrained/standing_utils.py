import pinocchio as pin
import numpy as np
import meshcat


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

def get_solution_trajectories(solver, rmodel, rdata, supportFeetIds):
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
        force_list = []
        ct_frame_name = rmodel.frames[frame_idx].name + "_contact"
        datas = [solver.problem.runningDatas[i].differential.multibody.contacts.contacts[ct_frame_name] for i in range(N-1)]
        lwaMf = [solver.problem.runningDatas[i].differential.pinocchio.oMf[frame_idx].copy() for i in range(N-1)]
        for m in lwaMf:
            m.translation = np.zeros(3)
        ee_forces = [lwaMf[k].act(datas[k].jMf.actInv(datas[k].f)).vector for k in range(N-1)] 
        sol[ct_frame_name] = [ee_forces[i] for i in range(N-1)]     
    
    return sol    
