from pytissueoptics.rayscattering.opencl.CLSource import CLPencilSource
from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.scene import Vector, Logger, MayaviViewer

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
source = CLPencilSource(position=Vector(0, 0, 0), direction=Vector(0, 0, 1), N=1000)

source.propagate(worldMaterial=worldMaterial, logger=logger)

viewer = MayaviViewer()
viewer.addLogger(logger)
viewer.show()
