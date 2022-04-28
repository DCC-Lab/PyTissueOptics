from pytissueoptics.rayscattering import *
from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.rayscattering.tissues import CubeTissue
from pytissueoptics.scene import Vector, Logger
import numpy as np

np.random.seed(15)

tissue = CubeTissue(material=ScatteringMaterial(2, 0.1, 0.99, 1.4))
source = PencilSource(position=Vector(0, 0, 0),
                      direction=Vector(0, 0, 1), N=20000)
logger = Logger()

source.propagate(tissue, logger=logger)
stats = Stats(logger, source, tissue)


stats.showEnergy2D(bins=51, logScale=False)

