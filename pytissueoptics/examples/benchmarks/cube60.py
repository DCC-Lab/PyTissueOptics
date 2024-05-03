import numpy as np

import env
from pytissueoptics import *

TITLE = "MCX Homogeneous cube"

DESCRIPTION = """ Pencil source propagation through a homogeneous cube of size 60x60x60 mm. """


def exampleCode():
    N = 100000 if hardwareAccelerationIsAvailable() else 1000
    absorbancess = []
    for ua in np.linspace(0.001, 0.05, 50):
        tissue = ScatteringScene([Cube(60, material=ScatteringMaterial(mu_a=ua, mu_s=1, g=0.01, n=1), label="cube")])
        logger = EnergyLogger(tissue, defaultBinSize=0.1)
        source = PencilPointSource(position=Vector(0, 0, -29.99), direction=Vector(0, 0, 1), N=N, displaySize=1)

        source.propagate(tissue, logger=logger)

        stats = Stats(logger)

        absorbancess.append((ua, stats.getAbsorbance("cube")))
        #viewer.reportStats()

        #viewer.show2D(View2DProjectionX())
        #viewer.show1D(Direction.Z_POS)
        #viewer.show3D(pointCloudStyle=PointCloudStyle(showSolidPoints=False))
        #viewer.show3D()
    print(*absorbancess, sep="\n")


if __name__ == "__main__":
    exampleCode()
