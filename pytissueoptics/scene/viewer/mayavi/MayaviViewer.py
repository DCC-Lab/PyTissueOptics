import numpy as np

from pytissueoptics.scene.logger import Logger
from pytissueoptics.scene.scene import Scene

try:
    from mayavi import mlab
    MAYAVI_AVAILABLE = True
except ImportError:
    MAYAVI_AVAILABLE = False

from pytissueoptics.scene.viewer.mayavi import MayaviSolid
from pytissueoptics.scene.solids import Solid


class MayaviViewer:
    def __init__(self):
        self._scenes = {
            "DefaultScene": {"figureParameters": {"bgColor": (0.11, 0.11, 0.11), "fgColor": (0.9, 0.9, 0.9)},
                             "Solids": [], }}
        self._view = {"azimuth": -30, "zenith": 215, "distance": None, "pointingTowards": None, "roll": -0}
        self.clear()

    def addScene(self, scene: Scene, representation="wireframe", lineWidth=0.25, showNormals=False, normalLength=0.3,
            colormap="viridis", reverseColormap=False, constantColor=False, opacity=1, **kwargs):
        self.add(*scene.solids, representation=representation, lineWidth=lineWidth, showNormals=showNormals,
                 normalLength=normalLength, colormap=colormap, reverseColormap=reverseColormap,
                 constantColor=constantColor, opacity=opacity, **kwargs)

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
    def addPoints(points: np.ndarray, colormap="rainbow", reverseColormap=False, scale=0.01, asSpheres=True):
        """ 'points' has to be of shape (n, 3) where the second axis is (x, y, z). """
        if points is None:
            return
        x, y, z = [points[:, i] for i in range(3)]
        mode = "sphere" if asSpheres else "point"
        s = mlab.points3d(x, y, z, mode=mode, scale_factor=scale, scale_mode="none", colormap=colormap)
        s.module_manager.scalar_lut_manager.reverse_lut = reverseColormap

    @staticmethod
    def addDataPoints(dataPoints: np.ndarray, colormap="rainbow", reverseColormap=False, scale=0.15,
                      scaleWithValue=True, asSpheres=True):
        """ 'dataPoints' has to be of shape (n, 4) where the second axis is (value, x, y, z). """
        if dataPoints is None:
            return
        v, x, y, z = [dataPoints[:, i] for i in range(4)]
        scaleMode = "scalar" if scaleWithValue else "none"
        mode = "sphere" if asSpheres else "point"
        s = mlab.points3d(x, y, z, v, mode=mode, scale_factor=scale, scale_mode=scaleMode, colormap=colormap)
        s.module_manager.scalar_lut_manager.reverse_lut = reverseColormap

    @staticmethod
    def addSegments(segments: np.ndarray, colormap="rainbow", reverseColormap=False):
        """ 'segments' has to be of shape (n, 6) where the second axis is (x1, y1, z1, x2, y2, z2). """
        if segments is None:
            return
        for segment in segments:
            x = [segment[0], segment[3]]
            y = [segment[1], segment[4]]
            z = [segment[2], segment[5]]
            s = mlab.plot3d(x, y, z, tube_radius=None, line_width=1, colormap=colormap)
            s.module_manager.scalar_lut_manager.reverse_lut = reverseColormap

    @staticmethod
    def addImage(image: np.ndarray, sizeInCM: tuple = None, minCorner: tuple = (0, 0),
                 axis: int = 2, position: float = 0):
        if sizeInCM is None:
            sizeInCM = image.shape
        overSampling = 5  # 10% lost on edge pixel (0.5/oversampling)
        image = np.repeat(np.repeat(image, overSampling, axis=0), overSampling, axis=1)

        image = np.flip(image, axis=0)
        image = np.flip(image, axis=1)

        p = mlab.imshow(image, colormap='viridis', interpolate=False,
                        extent=[0, sizeInCM[0], 0, sizeInCM[1], position, position], )
        p.actor.force_opaque = True

        tempPosition = [minCorner[0] + sizeInCM[0] / 2, minCorner[1] + sizeInCM[1] / 2]
        tempPosition.insert(axis, position)
        p.actor.position = tempPosition

        if axis == 0:
            p.actor.rotate_y(90)
        elif axis == 1:
            p.actor.rotate_x(90)
            p.actor.rotate_z(-90)
        return p

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
    from pytissueoptics.scene import Sphere, Cuboid, Vector, Scene

    sphere1 = Sphere(order=2)
    cuboid1 = Cuboid(1, 3, 3, position=Vector(4, 0, 0))
    viewer = MayaviViewer()
    viewer.add(sphere1, cuboid1, lineWidth=1, showNormals=True)
    viewer.show()
