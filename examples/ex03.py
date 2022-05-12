from pytissueoptics import *

TITLE = "Propagate in a custom scene and play with focal of different objects."

DESCRIPTION = """  
There are Cuboid() which serve as a screens for visualization, and an Ellipsoid() as a lens. They all go into a RayScatteringScene
which takes a list of solid. We can use the MayaviViewer to view our scene before propagation. Then, we repeat the
usual steps of propagation. By changing the index 'n' of the lens material, we can see how the focal is affected.
"""


def exampleCode():
    diffusiveMaterial = ScatteringMaterial(mu_s=0.0, mu_a=0, g=0.7, n=1.34)
    absorptiveMaterial = ScatteringMaterial(mu_s=1.0, mu_a=0.5, g=1.0)

    screen = Cuboid(a=0.1, b=4, c=4, position=Vector(10, 0, 0), material=absorptiveMaterial)
    screen2 = Cuboid(a=0.1, b=4, c=4, position=Vector(5, 0, 0), material=absorptiveMaterial)
    screen3 = Cuboid(a=0.1, b=4, c=4, position=Vector(3, 0, 0), material=absorptiveMaterial)

    ellipsoid = Ellipsoid(a=0.5, b=2, c=2, position=Vector(-1, 0, 0), material=diffusiveMaterial)
    myCustomScene = RayScatteringScene([screen, screen2, screen3, ellipsoid])

    viewer = MayaviViewer()
    viewer.addScene(myCustomScene)
    viewer.show()

    logger = Logger()
    source = DirectionalSource(position=Vector(-3, 0, 0), direction=Vector(1, 0, 0), radius=1, N=10000)
    source.propagate(myCustomScene, logger)

    stats = Stats(logger, source, myCustomScene)
    stats.showEnergy3D()


if __name__ == "__main__":
    exampleCode()
