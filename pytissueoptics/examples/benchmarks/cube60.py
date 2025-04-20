import env  # noqa: F401
from pytissueoptics import *  # noqa: F403

TITLE = "MCX Homogeneous cube"

DESCRIPTION = """ Pencil source propagation through a homogeneous cube of size 60x60x60 mm. """


def exampleCode():
    N = 100000 if hardwareAccelerationIsAvailable() else 1000

    tissue = ScatteringScene([Cube(60, material=ScatteringMaterial(mu_a=0.005, mu_s=1, g=0.01, n=1))])
    logger = EnergyLogger(tissue, defaultBinSize=0.1)
    source = PencilPointSource(position=Vector(0, 0, -29.99), direction=Vector(0, 0, 1), N=N, displaySize=1)

    source.propagate(tissue, logger=logger)

    viewer = Viewer(tissue, source, logger)
    viewer.reportStats()

    viewer.show2D(View2DProjectionX())
    viewer.show1D(Direction.Z_POS)
    viewer.show3D(pointCloudStyle=PointCloudStyle(showSolidPoints=False))
    viewer.show3D()


if __name__ == "__main__":
    exampleCode()
