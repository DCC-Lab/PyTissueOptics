from pytissueoptics.rayscattering import *
from pytissueoptics.rayscattering import Stats
from pytissueoptics.rayscattering.tissues import PhantomTissue
from pytissueoptics.scene import Vector, Logger
import numpy as np

np.random.seed(15)

tissue = PhantomTissue()
source = PencilSource(position=Vector(0, 0, -1), direction=Vector(0, 0, 1), N=1000)
logger = Logger()

source.propagate(tissue, logger=logger)

stats = Stats(logger, tissue)
stats.showEnergy3D()
stats.showEnergy3DOfSurfaces()
