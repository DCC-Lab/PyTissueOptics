import numpy as np
from mayavi import mlab

from pytissueoptics.scene.geometry import BoundingBox
from pytissueoptics.scene.logger import Logger
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.viewer import Abstract3DViewer, ViewPointStyle
from pytissueoptics.scene.viewer.viewPoint import ViewPointFactory

from .mayaviSolid import MayaviSolid


class Mayavi3DViewer(Abstract3DViewer):
    def __init__(self):
        self._scenes = {
            "DefaultScene": {
                "figureParameters": {"bgColor": (0.11, 0.11, 0.11), "fgColor": (0.9, 0.9, 0.9)},
                "Solids": [],
            }
        }
        self._viewPoint = ViewPointFactory().create(ViewPointStyle.NATURAL)
        self.clear()

    def setViewPointStyle(self, viewPointStyle: ViewPointStyle):
        self._viewPoint = ViewPointFactory().create(viewPointStyle)

    def add(
        self,
        *solids: "Solid",
        representation="wireframe",
        lineWidth=0.25,
        showNormals=False,
        normalLength=0.3,
        colormap="viridis",
        reverseColormap=False,
        colorWithPosition=False,
        opacity=1,
        **kwargs,
    ):
        for solid in solids:
            mayaviSolid = MayaviSolid(solid, loadNormals=showNormals)
            self._scenes["DefaultScene"]["Solids"].append(mayaviSolid)
            s = mlab.triangular_mesh(
                *mayaviSolid.triangleMesh.components,
                representation=representation,
                line_width=lineWidth,
                colormap=colormap,
                opacity=opacity,
                **kwargs,
            )
            s.module_manager.scalar_lut_manager.reverse_lut = reverseColormap
            if colorWithPosition:
                s.module_manager.lut_data_mode = "cell data"
            if showNormals:
                mlab.quiver3d(
                    *mayaviSolid.normals.components, line_width=lineWidth, scale_factor=normalLength, color=(1, 1, 1)
                )

    def addLogger(
        self,
        logger: Logger,
        colormap="rainbow",
        reverseColormap=False,
        pointScale=0.01,
        dataPointScale=0.15,
        scaleWithValue=True,
    ):
        self.addPoints(logger.getPoints(), colormap=colormap, reverseColormap=reverseColormap, scale=pointScale)
        self.addDataPoints(
            logger.getRawDataPoints(),
            colormap=colormap,
            reverseColormap=reverseColormap,
            scale=dataPointScale,
            scaleWithValue=scaleWithValue,
        )
        self.addSegments(logger.getSegments(), colormap=colormap, reverseColormap=reverseColormap)

    @staticmethod
    def addPoints(points: np.ndarray, colormap="rainbow", reverseColormap=False, scale=0.01, asSpheres=True):
        """'points' has to be of shape (n, 3) where the second axis is (x, y, z)."""
        if points is None:
            return
        x, y, z = [points[:, i] for i in range(3)]
        mode = "sphere" if asSpheres else "point"
        s = mlab.points3d(x, y, z, mode=mode, scale_factor=scale, scale_mode="none", colormap=colormap)
        s.module_manager.scalar_lut_manager.reverse_lut = reverseColormap

    def addDataPoints(
        self,
        dataPoints: np.ndarray,
        colormap="rainbow",
        reverseColormap=False,
        scale=0.15,
        scaleWithValue=True,
        asSpheres=True,
    ):
        """'dataPoints' has to be of shape (n, 4) where the second axis is (value, x, y, z)."""
        if dataPoints is None:
            return
        v, x, y, z = [dataPoints[:, i] for i in range(4)]
        scaleMode = "scalar" if scaleWithValue else "none"
        mode = "sphere" if asSpheres else "point"
        s = mlab.points3d(x, y, z, v, mode=mode, scale_factor=scale, scale_mode=scaleMode, colormap=colormap)
        s.module_manager.scalar_lut_manager.reverse_lut = reverseColormap

    @staticmethod
    def addSegments(segments: np.ndarray, colormap="rainbow", reverseColormap=False):
        """'segments' has to be of shape (n, 6) where the second axis is (x1, y1, z1, x2, y2, z2)."""
        if segments is None:
            return
        for segment in segments:
            x = [segment[0], segment[3]]
            y = [segment[1], segment[4]]
            z = [segment[2], segment[5]]
            s = mlab.plot3d(x, y, z, tube_radius=None, line_width=1, colormap=colormap)
            s.module_manager.scalar_lut_manager.reverse_lut = reverseColormap

    def addImage(
        self,
        image: np.ndarray,
        size: tuple = None,
        minCorner: tuple = (0, 0),
        axis: int = 2,
        position: float = 0,
        colormap: str = "viridis",
    ):
        if size is None:
            size = image.shape
        # Limitation: the current call to Mayavi.mlab.imshow will display half of the first pixel.
        # Workaround: oversample the image.
        #  However, this can become expensive for large images where the edge is not that visible anyway,
        #  so we use less oversampling.
        overSampling = 5  # 10% lost on edge pixel (0.5/oversampling)
        if image.size > 10**6:
            overSampling = 2  # 25% lost on edge pixel for images > 1MB
        if image.size > 10**7:
            overSampling = 1  # 50% lost on edge pixel for images > 10MB
        image = np.repeat(np.repeat(image, overSampling, axis=0), overSampling, axis=1)

        # In the 3D viewer, right is negative X and down is negative Y.
        #  This is the opposite what is expected by mlab.imshow.
        image = np.flip(image, axis=0)
        image = np.flip(image, axis=1)

        # The (X, Y) size has to be flipped to match rotations below for image axis 0 and 1.
        displaySize = size if axis == 2 else size[::-1]

        p = mlab.imshow(
            image,
            colormap=colormap,
            interpolate=False,
            extent=[0, displaySize[0], 0, displaySize[1], position, position],
        )
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

    def save(self, filepath):
        self._assignViewPoint()
        mlab.savefig(filepath, magnification=1)

    def _resetTo(self, scene):
        figParams = self._scenes[scene]["figureParameters"]
        bgColor = figParams["bgColor"]
        fgColor = figParams["fgColor"]
        fig = mlab.gcf()
        mlab.figure(figure=fig, bgcolor=bgColor, fgcolor=fgColor)

    def clear(self):
        mlab.clf()
        self._resetTo("DefaultScene")

    def close(self):
        mlab.close()

    def addBBox(self, bbox: BoundingBox, lineWidth=0.25, color=(1, 1, 1), opacity=1.0, **kwargs):
        """Adds a bounding box to the scene."""
        s = mlab.plot3d(
            [bbox.xMin, bbox.xMax],
            [bbox.yMin, bbox.yMax],
            [bbox.zMin, bbox.zMax],
            tube_radius=None,
            line_width=0,
            opacity=0,
        )
        mlab.outline(s, line_width=lineWidth, color=color, opacity=opacity, **kwargs)

    @staticmethod
    def showVolumeSlicer(hist3D: np.ndarray, colormap="viridis", interpolate=False, **kwargs):
        from .mayaviVolumeSlicer import MayaviVolumeSlicer

        slicer = MayaviVolumeSlicer(hist3D, colormap=colormap, interpolate=interpolate, **kwargs)
        slicer.show()
