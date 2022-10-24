from pytissueoptics import *

material1 = ScatteringMaterial(mu_s=1.5, mu_a=1, g=0.8, n=1.4)
material2 = ScatteringMaterial(mu_s=2.5, mu_a=1, g=0.8, n=1.7)

cube = Cuboid(a=3, b=3, c=3, position=Vector(0, 0, 1.5), material=material1, label="Cube")

sphere = Sphere(radius=1, order=3, position=Vector(0, 0, 1.5), material=material2, label="Sphere",
                smooth=True)

layerTissueScene = RayScatteringScene([cube, sphere])

N = 500000
logger = Logger()
source = PencilPointSource(position=Vector(0, 0, -1), direction=Vector(0, 0, 1), N=N,
                           useHardwareAcceleration=True)
source.propagate(layerTissueScene, logger)

stats = Stats(logger, source, layerTissueScene)
stats.report()

# stats.showEnergy3D(config=DisplayConfig(showPointsAsSpheres=False))

