try:
    from mayavi import mlab
except ImportError:
    pass

from pytissueoptics.scene.viewer.mayavi import MayaviSolid
from pytissueoptics.scene.solids import Sphere, Cuboid
from pytissueoptics.scene.geometry import Vector, primitives


class MayaviViewer:
    def __init__(self):
        self._scenes = {"DefaultScene": {"figure": mlab.figure(1, bgcolor=(0.11, 0.11, 0.11), fgcolor=(0.9, 0.9, 0.9)), "Solids": [], }}
        self._view = {"azimuth": 0, "zenith": 0, "distance": None, "pointingTowards": None, "roll": None}

    def addMayaviSolid(self, other: 'MayaviSolid', representation="wireframe", line_width=0.25):
        assert other.primitive == primitives.TRIANGLE, "MavaviViewer currently only supports triangle mesh. "
        self._scenes["DefaultScene"]["Solids"].append(other)
        mlab.triangular_mesh(*other.mesh.components, representation=representation, line_width=line_width, colormap="viridis")

    def _assignViewPoint(self):
        azimuth, elevation, distance, towards, roll = (self._view[key] for key in self._view)
        mlab.view(azimuth, elevation, distance, towards, roll)

    def show(self):
        self._assignViewPoint()
        mlab.show()


if __name__ == "__main__":
    sphere1 = MayaviSolid(Sphere(a=3, b=1, c=0.6, order=4))
    viewer = MayaviViewer()
    viewer.addMayaviSolid(sphere1, representation="surface")

    viewer.show()
