import os
import sys

modulePath = os.path.abspath(__file__ + 4 * '/..')
sys.path.append(modulePath)

from pytissueoptics import *
P = 3  # 1st lens position
r = 2.0  # ellipsoid a & b radii

# Approximate focal lengths
F1 = 9.87
F2 = 19 - 3

H = 8
L = P * 2 + F1 + F2

space = Cuboid(H, H, L, material=ScatteringMaterial(mu_s=5, mu_a=0.05, g=1, n=1), label="space")
space.translateTo(Vector(0, 0, L/2))
lens1 = Ellipsoid(r, r, 0.5, position=Vector(0, 0, P), material=ScatteringMaterial(mu_s=5, mu_a=0.05, g=1, n=1.4))
lens2 = Ellipsoid(r, r, 0.25, position=Vector(0, 0, P + F1 + F2), material=ScatteringMaterial(mu_s=5, mu_a=0.05, g=1, n=1.4))

scene = ScatteringScene([space, lens1, lens2])
logger = EnergyLogger(scene)
source = DirectionalSource(position=Vector(0, 0, -2), direction=Vector(0, 0, 1), diameter=1,
                           N=50000, useHardwareAcceleration=True)

source.propagate(scene, logger)

viewer = Viewer(scene, source, logger)
viewer.reportStats()
viewer.show2D(View2DProjectionY())
viewer.show3D()
