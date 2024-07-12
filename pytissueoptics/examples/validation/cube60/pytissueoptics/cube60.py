import numpy as np

from pytissueoptics import *
import matplotlib.pyplot as plt

TITLE = "MCX Homogeneous cube"

DESCRIPTION = """ Pencil source propagation through a homogeneous cube of size 60x60x60 mm. """



def exampleCode():
    N = 100000 if hardwareAccelerationIsAvailable() else 10000
    absorbancess = []
    parameters = []
    for mus in np.logspace(0.1, 20, 30, base=1.5)/100:
        np.random.seed(0)
        tissue = ScatteringScene([Cube(60, material=ScatteringMaterial(mu_a=0.005, mu_s=mus, g=0.8, n=1), label="cube")])
        logger = EnergyLogger(tissue, defaultBinSize=0.1)
        source = PencilPointSource(position=Vector(0, 0, -29.99), direction=Vector(0, 0, 1), N=N, displaySize=1)

        source.propagate(tissue, logger=logger)

        stats = Stats(logger)

        absorbancess.append((stats.getAbsorbance("cube")))
        parameters.append(mus)

   #viewer = Viewer(tissue, logger=logger, source=source)
    #viewer.reportStats()
    #viewer.show2D(View2DProjectionX())
        #viewer.show1D(Direction.Z_POS)
        #viewer.show3D(pointCloudStyle=PointCloudStyle(showSolidPoints=False))
        #viewer.show3D()

    #plt.plot(parameters, absorbancess)
    #plt.show()
    print(*parameters, *absorbancess, sep="\n")


if __name__ == "__main__":
    exampleCode()
