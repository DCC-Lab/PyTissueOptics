from pytissueoptics import *


TITLE = "Propagate in a custom scene"

DESCRIPTION = """  
There is a Cuboid(), a Sphere() and an Ellipsoid(). They all go into a RayScatteringScene which takes a list of solid.
We can use the MayaviViewer to view our scene before propagation. Then, we repeat the usual steps of propagation.
"""


def exampleCode():
    air = ScatteringMaterial(mu_s=0, mu_a=0, g=1.0)
    water = ScatteringMaterial(mu_s=0.005, mu_a=0.002, g=0.9)

    airLayer = Cuboid(a=5000, b=5000, c=1000, position=Vector(0, 0, 6), material=air)
    waterLayer = Cuboid(a=5000, b=5000, c=1000, position=Vector(0, 0, 0), material=water)
    cuboidStack = airLayer.stack(waterLayer, onSurface="front")
    myCustomScene = RayScatteringScene(solids=[cuboidStack])

    logger = Logger()
    source = PencilSource(position=Vector(0, 0, -10), direction=Vector(0, 0, 1), N=1000)
    source.propagate(myCustomScene, logger)

    stats = Stats(logger, source, myCustomScene)
    stats.showEnergy3D()


if __name__ == "__main__":
    exampleCode()
