from pytissueoptics import *

TITLE = "Propagation in an imported .obj 3d model."

DESCRIPTION = """  
This example shows how to propagate photons in an imported .obj 3d model.
Materials in the 3d models are not imported by default, so you need to
give objects a material before you can propagate photons in them.
"""


def exampleCode():
    material = ScatteringMaterial(mu_s=5.0, mu_a=1.0, g=0.9)

    toyModel = loadSolid("exampleFile.obj", position=Vector(0, 0, 0), material=material)
    myCustomScene = RayScatteringScene([toyModel])

    logger = Logger()
    source = IsotropicPointSource(position=Vector(-1, 4, 0), N=1000)
    source.propagate(myCustomScene, logger)

    stats = Stats(logger, source, myCustomScene)
    stats.showEnergy3D()
    stats.showEnergy2D(projection='z')
    stats.report()


if __name__ == "__main__":
    exampleCode()
