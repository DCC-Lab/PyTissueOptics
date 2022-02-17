try:
    from mayavi import mlab
except ImportError:
    pass

from pytissueoptics.scene.viewer.mayavi import MayaviSolid
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.geometry import primitives


class MayaviViewer:
    def __init__(self):
        self._scenes = {"DefaultScene": {"figureParameters": {"bgColor": (0.11, 0.11, 0.11), "fgColor": (0.9, 0.9, 0.9)}, "Solids": [], }}
        self._view = {"azimuth": 0, "zenith": 0, "distance": None, "pointingTowards": None, "roll": None}
        self.clear()

    def add(self, *solids: 'Solid', representation="wireframe", lineWidth=0.25, showNormals=False, normalLength=0.3):
        for solid in solids:
            mayaviSolid = MayaviSolid(solid, loadNormals=showNormals)
            self._scenes["DefaultScene"]["Solids"].append(mayaviSolid)
            self._createPolygonMesh(mayaviSolid.mesh.components, representation, lineWidth,
                                    colormap="viridis", primitive=solid.primitive)
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

    @staticmethod
    def _createPolygonMesh(meshComponents, representation, lineWidth, colormap, primitive):
        if primitive == primitives.QUAD:
            x, y, z, quadIndices = meshComponents
            for indices in quadIndices:
                a, b, c, d = indices
                xp = [[x[a], x[d]], [x[b], x[c]]]
                yp = [[y[a], y[d]], [y[b], y[c]]]
                zp = [[z[a], z[d]], [z[b], z[c]]]
                mlab.mesh(xp, yp, zp, representation=representation, line_width=lineWidth,
                          colormap=colormap)
        else:
            mlab.triangular_mesh(*meshComponents, representation=representation, line_width=lineWidth,
                                 colormap="viridis")


if __name__ == "__main__":
    from pytissueoptics.scene import Sphere, Cuboid, Vector

    sphere1 = Sphere(order=2)
    cuboid1 = Cuboid(1, 3, 3, position=Vector(4, 0, 0))
    viewer = MayaviViewer()
    viewer.add(sphere1, cuboid1, lineWidth=1, showNormals=True)
    viewer.show()
