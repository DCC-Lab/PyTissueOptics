from pytissueoptics.rayscattering import *
from pytissueoptics.rayscattering.tissues import PhantomTissue
from pytissueoptics.scene import Vector, Logger, MayaviViewer
import numpy as np

np.random.seed(15)

tissue = PhantomTissue()
source = PencilSource(position=Vector(0, 0, -1), direction=Vector(0, 0, 1), N=2000)
logger = Logger()

source.propagate(tissue, logger=logger)

viewer = MayaviViewer()
tissue.addToViewer(viewer)
viewer.addLogger(logger)
viewer.show()
