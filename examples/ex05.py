from pytissueoptics import *

TITLE = "Propagation in an imported .obj 3d model."

DESCRIPTION = """  
This example shows how to propagate photons in an imported .obj 3d model.
Materials in the 3d models are not imported by default, so you need to
give objects a material before you can propagate photons in them.
"""


def exampleCode():
    material = ScatteringMaterial(mu_s=0.0, mu_a=0.0, g=1.0, n=1.2)
    absorptiveMaterial = ScatteringMaterial(mu_s=0.0, mu_a=10.0, g=1.0)

    toyModel = loadSolid("trisphere.obj", position=Vector(0, 0, 0), material=material, smooth=True)
    toyModel.scale(0.3)
    toyModel.rotate(0, 15, 0)
    screen1 = Cuboid(a=0.1, b=30, c=30, position=Vector(15, 0, 0), material=absorptiveMaterial)
    screen2 = Cuboid(a=0.1, b=30, c=30, position=Vector(15, 0, 0), material=absorptiveMaterial)
    screen3 = Cuboid(a=0.1, b=30, c=30, position=Vector(15, 0, 0), material=absorptiveMaterial)
    screen2.rotate(0, 90, 0, rotationCenter=Vector(0, 0, 0))
    screen2.translateBy(Vector(0, 0, -0.1))
    screen3.rotate(0, -90, 0, rotationCenter=Vector(0, 0, 0))
    screen3.translateBy(Vector(0, 0, 0.1))
    myCustomScene = RayScatteringScene([toyModel, screen1, screen2, screen3])

    logger = Logger()
    source = DirectionalSource(position=Vector(-10, 0, 0), direction=Vector(1, 0, 0), diameter=3, N=20000)
    source.propagate(myCustomScene, logger)

    stats = Stats(logger, source, myCustomScene)
    stats.showEnergy3D()
    stats.showEnergy2D(projection='z')
    stats.report()


if __name__ == "__main__":
    exampleCode()
