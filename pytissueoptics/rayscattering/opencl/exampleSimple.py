import os
import sys

import numpy as np

modulePath = os.path.abspath(__file__ + 4 * '/..')
sys.path.append(modulePath)

from pytissueoptics import *


N = 500000

scene = tissues.PhantomTissue()
source = DivergentSource(position=Vector(0, 0, -0.1), direction=Vector(0, 0, 1), N=N,
                         useHardwareAcceleration=True, diameter=0.2, divergence=np.pi / 4)
logger = EnergyLogger(scene)

source.propagate(scene, logger=logger)

viewer = Viewer(scene, source, logger)
viewer.reportStats()

viewer.show2D(View2DProjectionX())
viewer.show2D(View2DProjectionX(solidLabel="middleLayer"))
viewer.show2D(View2DSurfaceZ(solidLabel="middleLayer", surfaceLabel="interface0"))
viewer.show1D(Direction.Z_POS)
viewer.show3D()
