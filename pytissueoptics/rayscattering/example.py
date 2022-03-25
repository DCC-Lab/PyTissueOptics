from pytissueoptics.rayscattering import *
from pytissueoptics.rayscattering.tissues import PhantomTissue
from pytissueoptics.scene import *
import numpy as np

np.random.seed(15)

viewer = MayaviViewer()
tissue = PhantomTissue()
tissue.addToViewer(viewer)

source = PencilSource(position=Vector(0, 0, -1), direction=Vector(0, 0, 1),
                      N=2000)

logger = Logger()
source.propagate(tissue, logger=logger)

viewer.addLogger(logger)
viewer.show()
