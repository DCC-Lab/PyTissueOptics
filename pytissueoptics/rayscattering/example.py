from pytissueoptics.rayscattering import *
from pytissueoptics.rayscattering.tissues import PhantomTissue
from pytissueoptics.scene import Vector, Logger
import numpy as np

np.random.seed(15)

tissue = PhantomTissue()
source = PencilPointSource(position=Vector(0, 0, -1),
                           direction=Vector(0, 0, 1), N=2000)
logger = EnergyLogger(tissue)

source.propagate(tissue, logger=logger)

viewer = Viewer(tissue, source, logger)
viewer.reportStats()

viewer.show3D()
viewer.show3D(pointCloudStyle=PointCloudStyle(showSolidPoints=False))
viewer.show2D(View2DProjectionX())
viewer.show2D(View2DProjectionX(solidLabel="middleLayer"))
viewer.show2D(View2DSurfaceZ(solidLabel="middleLayer", surfaceLabel="interface1", surfaceEnergyLeaving=False))
viewer.show1D(Direction.Z_POS)
