import os
import sys

modulePath = os.path.abspath(__file__ + 4 * '/..')
sys.path.append(modulePath)

from pytissueoptics import *

mu_s, mu_a, g = 10, 0.1, 1
w = 10
topLayer = Cuboid(w, 1, w, material=ScatteringMaterial(mu_s, mu_a, g, 1.1), label="topLayer")
middleLayer = Cuboid(w, 1, w, material=ScatteringMaterial(mu_s, mu_a, g, 1.6), label="middleLayer")
bottomLayer = Cuboid(w, 1, w, material=ScatteringMaterial(mu_s, mu_a, g, 1.1), label="bottomLayer")

stack = bottomLayer.stack(middleLayer).stack(topLayer)
stack.translateTo(Vector(0, -1.5, 0))
scene = ScatteringScene([stack])

source = PencilPointSource(position=Vector(-2, 1, 0), direction=Vector(1, -1, 0), N=100000, useHardwareAcceleration=True)
logger = EnergyLogger(scene)

source.propagate(scene, logger)

viewer = Viewer(scene, source, logger)
viewer.reportStats()
viewer.show2D(View2DProjection(projectionDirection=Direction.Z_NEG, horizontalDirection=Direction.X_POS,
                               binSize=0.005, limits=((-1.5, 1.5), (-3, 0))), logScale=True)
viewer.show3D()
