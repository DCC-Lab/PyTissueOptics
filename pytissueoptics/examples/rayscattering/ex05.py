import env
from pytissueoptics import *

TITLE = "Sphere inside a cube"

DESCRIPTION = """ Propagation of a directional light source through a highly scattering medium scene composed of a
sphere inside a cube. """


def exampleCode():
    N = 10000 if hardwareAccelerationIsAvailable() else 200

    material1 = ScatteringMaterial(mu_s=20, mu_a=0.1, g=0.9, n=1.4)
    material2 = ScatteringMaterial(mu_s=30, mu_a=0.2, g=0.9, n=1.7)

    cube = Cuboid(a=3, b=3, c=3, position=Vector(0, 0, 0), material=material1, label="cube")
    sphere = Sphere(radius=1, order=3, position=Vector(0, 0, 0), material=material2, label="sphere",
                    smooth=True)
    scene = ScatteringScene([cube, sphere])

    logger = EnergyLogger(scene)
    source = DirectionalSource(position=Vector(0, 0, -2), direction=Vector(0, 0, 1), N=N, diameter=0.5)

    source.propagate(scene, logger)

    viewer = Viewer(scene, source, logger)
    viewer.reportStats()

    viewer.show2D(View2DProjectionY())
    viewer.show2D(View2DProjectionY(solidLabel="sphere"))
    viewer.show3D()


if __name__ == "__main__":
    exampleCode()
