import env  # noqa: F401

from pytissueoptics import *  # noqa: F403

TITLE = (
    "Propagate in a in a non-scattering custom scene with an optical lens."
    "Learn to save and load your data so you don't have to simulate again."
)

DESCRIPTION = """  
Thin Cuboid solids are used as screens for visualization, and a SymmetricLens() as a lens. They all go into a 
RayScatteringScene which takes a list of solid. We can display our scene before propagation. Then, we repeat the
usual steps of propagation. You can experiment with different focal lengths and material. The symmetric thick lens will
automatically compute the required surface curvatures to achieve the desired focal length.

The logger can save data to a file. You can then use logger.load(filepath) or Logger(filepath) to load the data. 
At that point you can comment out the line ‘source.propagate()‘ if you don't want to simulate again. You can explore 
the different views and information the object Stats provides.
"""


def exampleCode():
    N = 1000000 if hardwareAccelerationIsAvailable() else 2000
    glassMaterial = ScatteringMaterial(n=1.50)
    vacuum = ScatteringMaterial()
    screen1 = Cuboid(a=40, b=40, c=0.1, position=Vector(0, 0, 30), material=vacuum, label="Screen1")
    screen2 = Cuboid(a=40, b=40, c=0.1, position=Vector(0, 0, 70), material=vacuum, label="Screen2")
    screen3 = Cuboid(a=40, b=40, c=0.1, position=Vector(0, 0, 100.05), material=vacuum, label="Screen3")

    lens = SymmetricLens(f=100, diameter=25.4, thickness=3.6, material=glassMaterial, position=Vector(0, 0, 0))
    myCustomScene = ScatteringScene([screen1, screen2, screen3, lens])
    source = DirectionalSource(position=Vector(0, 0, -20), direction=Vector(0, 0, 1), diameter=20.0, N=N, displaySize=5)

    myCustomScene.show(source)

    logger = EnergyLogger(myCustomScene, "ex03.log")
    # The following line can be commented out if you only want to load the previously saved data.
    # Or leave it in if you want to continue the simulation.
    source.propagate(myCustomScene, logger)

    viewer = Viewer(myCustomScene, source, logger)
    viewer.reportStats()

    viewer.show3D()
    viewer.show2D(View2DSurfaceZ("Screen1", "front", surfaceEnergyLeaving=False))
    viewer.show2D(View2DSurfaceZ("Screen2", "front", surfaceEnergyLeaving=False))
    viewer.show2D(View2DSurfaceZ("Screen3", "front", surfaceEnergyLeaving=False))


if __name__ == "__main__":
    exampleCode()
