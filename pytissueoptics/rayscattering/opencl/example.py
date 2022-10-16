from pytissueoptics import *

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


material1 = ScatteringMaterial(0.1, 0.8, 0.8, 1.4)
material2 = ScatteringMaterial(2, 0.8, 0.8, 1.2)

layer1 = Cuboid(a=10, b=10, c=2, position=Vector(0, 0, 0), material=material1, label="Layer 1")
layer2 = Cuboid(a=10, b=10, c=2, position=Vector(0, 0, 0), material=material2, label="Layer 2")
tissue = layer1.stack(layer2, "back")
scene = RayScatteringScene([tissue])

# scene = InfiniteTissue(material=material1)
source = PencilPointSource(position=Vector(0, 0, 0), direction=Vector(0, 0, 1), N=10000, useHardwareAcceleration=True)
logger = Logger()

source.propagate(scene, logger=logger)

stats = Stats(logger, source, scene)
stats.showEnergy2D(bins=101, logScale=True, limits=[[-10, 10], [-10, 10]])
