import env
from pytissueoptics import *

TITLE = "Custom layer stack"

DESCRIPTION = """ This example shows how to make a layer stack. You initialize multiple Cuboid() and stack them in a 
layer stack. You can do this using cuboid.stack(secondCuboid). The stacked face need to have the same size on both 
cuboids. """


def exampleCode():
    N = 100000 if hardwareAccelerationIsAvailable() else 500
    materialLayer0 = ScatteringMaterial(mu_s=0, mu_a=0, g=1, n=1)
    materialLayer1 = ScatteringMaterial(mu_s=2, mu_a=0, g=0.99, n=1)
    materialLayer2 = ScatteringMaterial(mu_s=0, mu_a=100, g=1, n=1)

    layer1 = Cuboid(a=10, b=10, c=0.1, position=Vector(0, 0, 100), material=materialLayer2, label="Detector")

    tissue = ScatteringScene([layer1], worldMaterial=materialLayer0)

    logger = EnergyLogger(tissue)
    source = PencilPointSource(position=Vector(0, 0, 0), direction=Vector(0, 0, 1), N=N, displaySize=0.5, useHardwareAcceleration=False)

    for photon in source.photons:
        photon.scatterBy(*materialLayer1.getScatteringAngles())

    source.propagate(tissue, logger)

    viewer = Viewer(tissue, source, logger)
    viewer.reportStats()
    viewer.show2D(View2DSurfaceZ("Detector", "front", surfaceEnergyLeaving=False))
    viewer.show1D(Direction.X_POS)


if __name__ == "__main__":
    exampleCode()
