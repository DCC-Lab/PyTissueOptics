import os
import sys

modulePath = os.path.abspath(__file__ + 4 * '/..')
sys.path.append(modulePath)

from pytissueoptics import *

HIGH_SCATTERING = True

if HIGH_SCATTERING:
    N = 1000

    material1 = ScatteringMaterial(mu_s=20, mu_a=0.1, g=0.9, n=1.0)
    material2 = ScatteringMaterial(mu_s=30, mu_a=0.2, g=0.9, n=1.7)
else:
    N = 2000000

    material1 = ScatteringMaterial(mu_s=0.01, mu_a=1, g=0, n=1.0)
    material2 = ScatteringMaterial(mu_s=0.001, mu_a=0.01, g=0.9, n=1.7)

cube = Cuboid(a=3, b=3, c=3, position=Vector(0, 0, 0), material=material1, label="cube")
sphere = Sphere(radius=1, order=3, position=Vector(0, 0, 0), material=material2, label="sphere",
                smooth=True)
scene = ScatteringScene([cube, sphere])

logger = EnergyLogger(scene)
source = DirectionalSource(position=Vector(0, 0, -2), direction=Vector(0, 0, 1), N=N,
                           useHardwareAcceleration=False, diameter=1)

source.propagate(scene, logger)

viewer = Viewer(scene, source, logger)
viewer.reportStats()

viewer.show2D(View2DProjectionY())
viewer.show2D(View2DProjectionY(solidLabel="sphere"))
viewer.show3D()
