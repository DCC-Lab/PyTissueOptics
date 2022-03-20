from pytissueoptics.rayscattering import *
from pytissueoptics.scene import *
import numpy as np

np.random.seed(15)

viewer = MayaviViewer()
tissue = tissues.PhantomTissue()
# tissue.display()

tissue.addToViewer(viewer)

logger = Logger()
source = Source(position=Vector(0, 0, -1), direction=Vector(0, 0, 1),
                nPhotons=100)

source.propagate(tissue, logger=logger)

viewer.addLogger(logger)
viewer.show()
