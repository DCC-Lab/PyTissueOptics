import env
from pytissueoptics import *

TITLE = "MCX Skin vessel"

DESCRIPTION = """ Adapted MCX built-in example - a 3-layer domain with a cylindrical vessel inclusion 
to simulate skin/vessel measurements. This benchmark was first constructed by Dr. Steve Jacques in his mcxyz software
"""


def exampleCode():
    N = 1000000 if hardwareAccelerationIsAvailable() else 100

    # Units in mm and mm-1
    waterLayer = Cuboid(1, 0.1, 1, material=ScatteringMaterial(mu_a=0.00004, mu_s=1, g=1, n=1.33), label="water")
    epidermisLayer = Cuboid(1, 0.06, 1, material=ScatteringMaterial(mu_a=1.65724, mu_s=37.59398, g=0.9, n=1.44), label="epidermis")
    dermisLayer = Cuboid(1, 0.84, 1, material=ScatteringMaterial(mu_a=0.04585, mu_s=35.65406, g=0.9, n=1.38), label="dermis")
    zStack = waterLayer.stack(epidermisLayer).stack(dermisLayer)
    zStack.translateTo(Vector(0, 0, 0))
    bloodVessel = Cylinder(0.1, 0.99, material=ScatteringMaterial(mu_a=23.05427, mu_s=9.3985, g=0.9, n=1.361), label="blood", smooth=True)

    scene = ScatteringScene([zStack, bloodVessel])

    source = DirectionalSource(position=Vector(0, -0.399, 0), direction=Vector(0, 1, 0), N=N, diameter=0.6, displaySize=0.06)
    logger = EnergyLogger(scene, defaultBinSize=0.001)
    source.propagate(scene, logger=logger)

    viewer = Viewer(scene, source, logger)
    viewer.reportStats()
    viewer.show2D(View2DProjectionX())
    viewer.show2D(View2DProjectionZ())
    viewer.show1D(Direction.Z_POS)
    viewer.show1D(Direction.Y_POS)

    # Displaying only the energy that crossed surfaces to save memory.
    viewer.show3D(pointCloudStyle=PointCloudStyle(showSolidPoints=False))

    viewer.show3DVolumeSlicer(0.005)


if __name__ == "__main__":
    exampleCode()
