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

