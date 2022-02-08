from typing import Union

try:
    from mayavi import mlab
except ImportError:
    pass

from pytissueoptics.scene.viewer.mayavi import MayaviSolid
from pytissueoptics.scene.solids import Sphere, Cuboid, Solid
from pytissueoptics.scene.geometry import Vector, primitives


class MayaviViewer:
    def __init__(self):
        self._scenes = {"DefaultScene": {"figure": mlab.figure(1, bgcolor=(0.11, 0.11, 0.11), fgcolor=(0.9, 0.9, 0.9)),
                                         "Solids": [], }}
        self._view = {"azimuth": 0, "zenith": 0, "distance": None, "pointingTowards": None, "roll": None}

    def add(self, *solids: 'Solid', representation="wireframe", lineWidth=0.25):
        for solid in solids:
            assert solid.primitive == primitives.TRIANGLE, "MavaviViewer currently only supports triangle mesh. "
            mayaviSolid = MayaviSolid(solid)
            self._scenes["DefaultScene"]["Solids"].append(mayaviSolid)
            mlab.triangular_mesh(*mayaviSolid.mesh.components, representation=representation, line_width=lineWidth,
                                 colormap="viridis")

    def _assignViewPoint(self):
        azimuth, elevation, distance, towards, roll = (self._view[key] for key in self._view)
        mlab.view(azimuth, elevation, distance, towards, roll)

    def show(self):
        self._assignViewPoint()
        mlab.show()


if __name__ == "__main__":
    sphere1 = Sphere(order=2)
    cuboid1 = Cuboid(1, 3, 3, position=Vector(4, 0, 0))
    viewer = MayaviViewer()
    viewer.show(sphere1, cuboid1)
