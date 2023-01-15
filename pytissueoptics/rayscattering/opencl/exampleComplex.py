from pytissueoptics import *

HIGH_SCATTERING = True

if HIGH_SCATTERING:
    N = 10000

    material1 = ScatteringMaterial(mu_s=20, mu_a=0.1, g=0.9, n=1.4)
    material2 = ScatteringMaterial(mu_s=30, mu_a=0.2, g=0.9, n=1.7)
else:
    N = 200000

    material1 = ScatteringMaterial(mu_s=1.5, mu_a=1, g=0.9, n=1.4)
    material2 = ScatteringMaterial(mu_s=2.5, mu_a=1, g=0.9, n=1.7)

cube = Cuboid(a=3, b=3, c=3, position=Vector(0, 0, 0), material=material1, label="Cube")
sphere = Sphere(radius=1, order=3, position=Vector(0, 0, 0), material=material2, label="Sphere",
                smooth=True)
scene = RayScatteringScene([cube, sphere])

logger = EnergyLogger(scene)
source = DirectionalSource(position=Vector(0, 0, -2), direction=Vector(0, 0, 1), N=N,
                           useHardwareAcceleration=True, diameter=0.5)

source.propagate(scene, logger)

viewer = Viewer(scene, source, logger)
viewer.reportStats()

viewer.show2D(View2DProjectionY())
viewer.show2D(View2DProjectionY(solidLabel="Sphere", limits=((-2, 2), (-2, 2))))
viewer.show3D()
