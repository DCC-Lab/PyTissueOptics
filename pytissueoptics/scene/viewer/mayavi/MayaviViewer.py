import numpy as np

from pytissueoptics.scene.logger import Logger

try:
    from mayavi import mlab
except ImportError:
    pass

from pytissueoptics.scene.viewer.mayavi import MayaviSolid
from pytissueoptics.scene.solids import Solid


class MayaviViewer:
    def __init__(self):
        self._scenes = {
            "DefaultScene": {"figureParameters": {"bgColor": (0.11, 0.11, 0.11), "fgColor": (0.9, 0.9, 0.9)},
                             "Solids": [], }}
        self._view = {"azimuth": 0, "zenith": 0, "distance": None, "pointingTowards": None, "roll": None}
        self.clear()

    def add(self, *solids: 'Solid', representation="wireframe", lineWidth=0.25, showNormals=False, normalLength=0.3,
            colormap="viridis", reverseColormap=False, constantColor=False, opacity=1, **kwargs):
        for solid in solids:
            mayaviSolid = MayaviSolid(solid, loadNormals=showNormals)
            self._scenes["DefaultScene"]["Solids"].append(mayaviSolid)
            s = mlab.triangular_mesh(*mayaviSolid.triangleMesh.components, representation=representation,
                                     line_width=lineWidth, colormap=colormap, opacity=opacity, **kwargs)
            s.module_manager.scalar_lut_manager.reverse_lut = reverseColormap
            if constantColor:
                s.module_manager.lut_data_mode = "cell data"
            if showNormals:
                mlab.quiver3d(*mayaviSolid.normals.components, line_width=lineWidth, scale_factor=normalLength,
                              color=(1, 1, 1))

    def addLogger(self, logger: Logger, colormap="rainbow", reverseColormap=False,
                  pointScale=0.01, dataPointScale=0.15, scaleWithValue=True):
        self.addPoints(logger.getPoints(), colormap=colormap, reverseColormap=reverseColormap, scale=pointScale)
        self.addDataPoints(logger.getDataPoints(), colormap=colormap, reverseColormap=reverseColormap,
                           scale=dataPointScale, scaleWithValue=scaleWithValue)
        self.addSegments(logger.getSegments(), colormap=colormap, reverseColormap=reverseColormap)

    @staticmethod
    def addPoints(points: np.ndarray, colormap="rainbow", reverseColormap=False, scale=0.01):
        """ 'points' has to be of shape (3, n) where the first axis is (x, y, z). """
        if points is None:
            return
        x, y, z = points
        s = mlab.points3d(x, y, z, mode="sphere", scale_factor=scale, scale_mode="none", colormap=colormap)
        s.module_manager.scalar_lut_manager.reverse_lut = reverseColormap

    @staticmethod
    def addDataPoints(dataPoints: np.ndarray, colormap="rainbow", reverseColormap=False, scale=0.15, scaleWithValue=True):
        """ 'dataPoints' has to be of shape (4, n) where the first axis is (value, x, y, z). """
        if dataPoints is None:
            return
        v, x, y, z = dataPoints
        scaleMode = "scalar" if scaleWithValue else "none"
        s = mlab.points3d(x, y, z, v, mode="sphere", scale_factor=scale, scale_mode=scaleMode, colormap=colormap)
        s.module_manager.scalar_lut_manager.reverse_lut = reverseColormap

    @staticmethod
    def addSegments(segments: np.ndarray, colormap="rainbow", reverseColormap=False):
        """ 'segments' has to be of shape (6, n) where the first axis is (x1, y1, z1, x2, y2, z2). """
        if segments is None:
            return
        for i in range(segments.shape[1]):
            x = [segments[0, i], segments[3, i]]
            y = [segments[1, i], segments[4, i]]
            z = [segments[2, i], segments[5, i]]
            s = mlab.plot3d(x, y, z, tube_radius=None, line_width=1, colormap=colormap)
            s.module_manager.scalar_lut_manager.reverse_lut = reverseColormap

    def _assignViewPoint(self):
        azimuth, elevation, distance, towards, roll = (self._view[key] for key in self._view)
        mlab.view(azimuth, elevation, distance, towards, roll)

    def show(self):
        self._assignViewPoint()
        mlab.show()

    def _resetTo(self, scene):
        figParams = self._scenes[scene]["figureParameters"]
        bgColor = figParams["bgColor"]
        fgColor = figParams["fgColor"]
        fig = mlab.gcf()
        mlab.figure(figure=fig, bgcolor=bgColor, fgcolor=fgColor)

    def clear(self):
        mlab.clf()
        self._resetTo("DefaultScene")


if __name__ == "__main__":
    from pytissueoptics.scene import Sphere, Cuboid, Vector

    sphere1 = Sphere(order=2)
    cuboid1 = Cuboid(1, 3, 3, position=Vector(4, 0, 0))
    viewer = MayaviViewer()
    viewer.add(sphere1, cuboid1, lineWidth=1, showNormals=True)
    viewer.show()
