import env
from pytissueoptics import *

TITLE = "Propagate in a custom scene and play with focal of different objects." \
        "Learn to save and load your data so you don't have to simulate again."

DESCRIPTION = """  
Thin Cuboid solids are used as screens for visualization, and an Ellipsoid() as a lens. They all go into a 
RayScatteringScene which takes a list of solid. We can display our scene before propagation. Then, we repeat the
usual steps of propagation. By changing the index 'n' of the lens material, we can see how the focal is affected.

The logger can save data to a file. You can then use logger.load(filepath) or Logger(filepath) to load the data. 
At that point you can comment out the line ‘source.propagate()‘ if you don't want to simulate again. You can explore 
the different views and information the object Stats provides.
"""


def exampleCode():
    N = 100000 if hardwareAccelerationIsAvailable() else 2000
    glassMaterial = ScatteringMaterial(mu_s=0.0, mu_a=0, g=0.7, n=1.34)
    absorptiveMaterial = ScatteringMaterial(mu_s=1.0, mu_a=0.5, g=1.0)
    blockMaterial = ScatteringMaterial(mu_s=1.0, mu_a=10, g=0.7, n=1.0)
    
    screen1 = Cuboid(a=4, b=4, c=0.1, position=Vector(0, 0, 3), material=absorptiveMaterial, label="Screen1")
    screen2 = Cuboid(a=4, b=4, c=0.1, position=Vector(0, 0, 5), material=absorptiveMaterial, label="Screen2")
    screen3 = Cuboid(a=4, b=4, c=0.1, position=Vector(0, 0, 10), material=blockMaterial, label="Screen3")

    ellipsoid = Ellipsoid(a=2, b=2, c=0.5, position=Vector(0, 0, -1), material=glassMaterial)
    myCustomScene = ScatteringScene([screen1, screen2, screen3, ellipsoid])
    source = DirectionalSource(position=Vector(0, 0, -3), direction=Vector(0, 0, 1), diameter=1, N=N, displaySize=0.5)

    myCustomScene.show(source=source)

    logger = EnergyLogger(myCustomScene)
    source.propagate(myCustomScene, logger)

    viewer = Viewer(myCustomScene, source, logger)
    viewer.reportStats()

    viewer.show3D()
    viewer.show2D(View2DProjectionZ("Screen1"), logScale=False)
    viewer.show2D(View2DSurfaceZ("Screen2", "front", surfaceEnergyLeaving=False), logScale=False)
    viewer.show2D(View2DProjectionZ("Screen3"))


if __name__ == "__main__":
    exampleCode()
