from pytissueoptics import *

TITLE = "MCX Skin vessel"

DESCRIPTION = """ Adapted MCX built-in example - a 200x200x200 3-layer domain with a cylindrical vessel inclusion 
to simulate skin/vessel measurements. This benchmark was first constructed by Dr. Steve Jacques in his mcxyz software
"""

# from mcxyz, the block is 1mm in size, not 200mm ...
# apply same correction to other sim? https://omlc.org/software/mc/mcxyz/


def exampleCode():
    import numpy as np
    np.random.seed(651)
    N = 10000 if hardwareAccelerationIsAvailable() else 100

    vessel = Cylinder(0.1, 0.999, position=Vector(0, 0, 0), material=ScatteringMaterial(mu_a=23.05, mu_s=9.398, g=0.9, n=1.37), label="vessel")
    bottomLayer = Cuboid(1, 0.1, 1, material=ScatteringMaterial(mu_a=0.00003564, mu_s=1, g=1, n=1.37), label="bottom")
    middleLayer = Cuboid(1, 0.06, 1, material=ScatteringMaterial(mu_a=1.657, mu_s=37.59, g=0.9, n=1.37), label="middle")
    topLayer = Cuboid(1, 0.84, 1, material=ScatteringMaterial(mu_a=0.0458, mu_s=35.65, g=0.9, n=1.37), label="top")
    zStack = bottomLayer.stack(middleLayer).stack(topLayer)
    zStack.translateTo(Vector(0, 0, 0))
    vessel.setOutsideEnvironment(zStack.getEnvironment("top_top"))
    scene = ScatteringScene([vessel, zStack], ignoreIntersections=True)

    source = DirectionalSource(position=Vector(0, -0.39, 0), direction=Vector(0, 1, 0), N=N, diameter=0.6)
    logger = EnergyLogger(scene, defaultBinSize=0.001)
    source.propagate(scene, logger=logger)

    viewer = Viewer(scene, source, logger)
    viewer.reportStats()
    viewer.show2D(View2DProjectionX())
    viewer.show2D(View2DProjectionZ())
    viewer.show1D(Direction.Z_POS)
    viewer.show3D()


if __name__ == "__main__":
    import env
    exampleCode()
