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

# FIXME: does not work with hardware acceleration. Rays are passing through the lens without intersecting it.


def exampleCode():
    # N = 100000 if hardwareAccelerationIsAvailable() else 2000
    N = 2000
    glassMaterial = ScatteringMaterial(mu_s=0.0, mu_a=0, g=0.7, n=1.34)
    absorptiveMaterial = ScatteringMaterial(mu_s=1.0, mu_a=0.5, g=1.0)
    blockMaterial = ScatteringMaterial(mu_s=1.0, mu_a=10, g=0.7, n=1.0)

    screen1 = Cuboid(a=0.1, b=4, c=4, position=Vector(3, 0, 0), material=absorptiveMaterial, label="Screen1")
    screen2 = Cuboid(a=0.1, b=4, c=4, position=Vector(5, 0, 0), material=absorptiveMaterial, label="Screen2")
    screen3 = Cuboid(a=0.1, b=4, c=4, position=Vector(10, 0, 0), material=blockMaterial, label="Screen3")

    ellipsoid = Ellipsoid(a=0.5, b=2, c=2, position=Vector(-1, 0, 0), material=glassMaterial)
    myCustomScene = ScatteringScene([screen1, screen2, screen3, ellipsoid])

    myCustomScene.display()

    logger = EnergyLogger(myCustomScene)
    source = DirectionalSource(position=Vector(-3, 0, 0), direction=Vector(1, 0, 0), diameter=1, N=N,
                               useHardwareAcceleration=False)
    source.propagate(myCustomScene, logger)

    viewer = Viewer(myCustomScene, source, logger)
    viewer.reportStats()

    viewer.show3D()
    viewer.show2D(View2DProjectionX("Screen1"), logScale=False)
    viewer.show2D(View2DSurfaceX("Screen2", "left", surfaceEnergyLeaving=False), logScale=False)
    viewer.show2D(View2DProjectionX("Screen3"))


if __name__ == "__main__":
    import env
    exampleCode()
