import numpy as np

from pytissueoptics.scene.logger import Logger
from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.viewer.mayavi.viewPoint import ViewPointStyle, ViewPointFactory


try:
    from mayavi import mlab
    MAYAVI_AVAILABLE = True
except ImportError:
    MAYAVI_AVAILABLE = False

from pytissueoptics.scene.viewer.mayavi import MayaviSolid
from pytissueoptics.scene.solids import Solid


class MayaviViewer:
    def __init__(self, viewPointStyle=ViewPointStyle.NATURAL):
        self._scenes = {
            "DefaultScene": {"figureParameters": {"bgColor": (0.11, 0.11, 0.11), "fgColor": (0.9, 0.9, 0.9)},
                             "Solids": [], }}
        self._viewPoint = ViewPointFactory().create(viewPointStyle)
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
    def addImage(image: np.ndarray, size: tuple = None, minCorner: tuple = (0, 0),
                 axis: int = 2, position: float = 0, colormap: str = 'viridis'):
        if size is None:
            size = image.shape
        overSampling = 5  # 10% lost on edge pixel (0.5/oversampling)
        image = np.repeat(np.repeat(image, overSampling, axis=0), overSampling, axis=1)

        # In the 3D viewer, right is negative X and down is negative Y.
        #  This is the opposite what is expected by mlab.imshow.
        image = np.flip(image, axis=0)
        image = np.flip(image, axis=1)

        # The (X, Y) size has to be flipped to match rotations below for image axis 0 and 1.
        displaySize = size if axis == 2 else size[::-1]

        p = mlab.imshow(image, colormap=colormap, interpolate=False,
                        extent=[0, displaySize[0], 0, displaySize[1], position, position])
        p.actor.force_opaque = True

        tempPosition = [minCorner[0] + size[0] / 2, minCorner[1] + size[1] / 2]
        tempPosition.insert(axis, position)
        p.actor.position = tempPosition

        if axis == 0:
            p.actor.rotate_y(90)
        elif axis == 1:
            p.actor.rotate_x(90)
            p.actor.rotate_z(-90)
        return p

    def _assignViewPoint(self):
        mlab.view(**self._viewPoint.__dict__)

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
