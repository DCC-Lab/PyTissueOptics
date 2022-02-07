try:
    from mayavi import mlab
except ImportError:
    pass
from typing import Union
from pytissueoptics.scene.viewer.mayavi import MayaviSolid
from pytissueoptics.scene.solids import Sphere, Cuboid, Solid
from pytissueoptics.scene.geometry import Vector, primitives
from pytissueoptics.scene.scene import Scene


class MayaviViewer:
    def __init__(self):
        self._scenes = {"DefaultScene": {"figure": mlab.figure(1, bgcolor=(0.11, 0.11, 0.11), fgcolor=(0.9, 0.9, 0.9)), "Solids": [], }}
        self._view = {"azimuth": 0, "zenith": 0, "distance": None, "pointingTowards": None, "roll": None}

    def add(self, other: 'Solid', representation="wireframe", line_width=0.25):
        assert other.primitive == primitives.TRIANGLE, "MavaviViewer currently only supports triangle mesh. "
        other = MayaviSolid(other)
        self._scenes["DefaultScene"]["Solids"].append(other)
        mlab.triangular_mesh(*other.mesh.components, representation=representation, line_width=line_width, colormap="viridis")

    def _assignViewPoint(self):
        azimuth, elevation, distance, towards, roll = (self._view[key] for key in self._view)
        mlab.view(azimuth, elevation, distance, towards, roll)

    def show(self, objectToDisplay: Union[Solid, Scene]):
        if isinstance(objectToDisplay, Solid):
            self.add(objectToDisplay)
        elif isinstance(objectToDisplay, Scene):
            for solidToDisplay in objectToDisplay.solids:
                self.add(solidToDisplay)

        self._assignViewPoint()
        mlab.show()


if __name__ == "__main__":
    sphere1 = Sphere(order=2)
    cuboid1 = Cuboid(1, 3, 3, position=Vector(4, 0, 0))
    scene1 = Scene([sphere1, cuboid1])
    viewer = MayaviViewer()
    viewer.show(scene1)
