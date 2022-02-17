try:
    from mayavi import mlab
except ImportError:
    pass

from pytissueoptics.scene.viewer.mayavi import MayaviSolid
from pytissueoptics.scene.solids import Solid


class MayaviViewer:
    def __init__(self):
        self._scenes = {"DefaultScene": {"figureParameters": {"bgColor": (0.11, 0.11, 0.11), "fgColor": (0.9, 0.9, 0.9)}, "Solids": [], }}
        self._view = {"azimuth": 0, "zenith": 0, "distance": None, "pointingTowards": None, "roll": None}
        self.clear()

    def add(self, *solids: 'Solid', representation="wireframe", lineWidth=0.25, showNormals=False, normalLength=0.3):
        for solid in solids:
            mayaviSolid = MayaviSolid(solid, loadNormals=showNormals)
            self._scenes["DefaultScene"]["Solids"].append(mayaviSolid)
            mlab.triangular_mesh(*mayaviSolid.triangleMesh.components, representation=representation, line_width=lineWidth,
                                 colormap="viridis")
            if showNormals:
                mlab.quiver3d(*mayaviSolid.normals.components, line_width=lineWidth, scale_factor=normalLength, color=(1, 1, 1))

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
