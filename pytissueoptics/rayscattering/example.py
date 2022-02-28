from pytissueoptics.rayscattering.source import Source
from pytissueoptics.rayscattering.tissues import PhantomTissue
from pytissueoptics.scene import MayaviViewer, Vector, Material
from pytissueoptics.scene.logger import Logger
import numpy as np

np.random.seed(11)

viewer = MayaviViewer()
tissue = PhantomTissue()
# tissue.display()

tissue.addToViewer(viewer)

logger = Logger()
source = Source(position=Vector(0, 0, -1), direction=Vector(0.4, 0, 1))

source.propagate(tissue, logger=logger)

viewer.addLogger(logger)
viewer.show()
