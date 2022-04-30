from pytissueoptics.rayscattering import *
from pytissueoptics.rayscattering.tissues import CubeTissue
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


tissue = CubeTissue(side=10, material=ScatteringMaterial(30, 0.1, 0.8, 1.4))
source = PencilSource(position=Vector(0, 0, 0), direction=Vector(0, 0, 1), N=50000, use_opencl=True)
logger = Logger()

source.propagate(tissue, logger=logger)

stats = Stats(logger, source, tissue)
stats.showEnergy2D(bins=101, logScale=True, range=[[-10, 10], [-10, 10]])
