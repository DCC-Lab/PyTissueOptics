import time

from pytissueoptics.rayscattering import *
from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.rayscattering.tissues import CubeTissue
from pytissueoptics.scene import Vector, Logger
import numpy as np

np.random.seed(15)

tissue = CubeTissue(side=10, material=ScatteringMaterial(30, 0.1, 0.8, 1.4))
source = PencilSource(position=Vector(0, 0, 0),
                      direction=Vector(0, 0, 1), N=20)
logger = Logger()

t0 = time.time_ns()
source.propagate(tissue, logger=logger)
print(f"Time: {(time.time_ns() - t0)/1e9} s")

stats = Stats(logger, source, tissue)


stats.showEnergy2D(bins=101, logScale=True, range=[[-10, 10], [-10, 10]])

