import numpy as np

from pytissueoptics import *

N = 500000

scene = tissues.PhantomTissue()

source = DivergentSource(position=Vector(0, 0, -0.1), direction=Vector(0, 0, 1), N=N,
                         useHardwareAcceleration=True, diameter=0.2, divergence=np.pi / 4)
logger = Logger()
source.propagate(scene, logger=logger)

stats = Stats(logger, source, scene)

stats.report()

# displayConfig = DisplayConfig(showPointsAsSpheres=False)
# stats.showEnergy3D(config=displayConfig)
stats.showEnergy2D(bins=101, logScale=True, limits=[[-1.5, 1.5], [0, 2]])

