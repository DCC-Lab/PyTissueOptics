import numpy as np

from pytissueoptics.rayscattering import Stats
from pytissueoptics.rayscattering.opencl.CLSource import CLPencilSource
from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.scene import Vector, Logger

"""
This example shows how to use the OpenCL Source.
For now, it is not possible to propagate photons in a complex scene, but only in a single material infinite scene.

From Jacques, Steven. (2013). Optical Properties of Biological Tissues: , it is possible to gather
the following information:
It shows that the maximum scattering length density (Mu_s) of biological tissues is around 30 - 40 cm-1.
The average absorption length density (Mu_a) is around 0.1 - 0.2 cm-1.
The average anisotropy coefficient is around 0.8 - 0.9

These parameters will be used to mimic the parameters a typical user would utilize in a simulation.
"""


worldMaterial = ScatteringMaterial(mu_s=30, mu_a=0.1, g=0.8, index=1.4)
logger = Logger()
source = CLPencilSource(position=Vector(0, 0, 0), direction=Vector(0, 0, 1), N=9900)

source.propagate(worldMaterial=worldMaterial, logger=logger)
# a = logger.getDataPoints()
# for i, data in enumerate(a):
#     for d in data:
#         if np.isnan(d) or np.isinf(d):
#             print(data)
stats = Stats(logger, source)
stats.showEnergy2D(bins=50, logScale=True)
