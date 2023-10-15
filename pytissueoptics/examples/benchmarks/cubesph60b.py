import env
from pytissueoptics import *

TITLE = "MCX Sphere"

DESCRIPTION = """ Pencil source propagation through a homogeneous cube of size 60x60x60 mm with a spherical absorbing inclusion in the center. """


def exampleCode():
    N = 100000 if hardwareAccelerationIsAvailable() else 1000

    cube = Cube(60, material=ScatteringMaterial(mu_a=0.005, mu_s=0, g=0.01, n=1.37))
    sphere = Sphere(15, order=3, material=ScatteringMaterial(mu_a=0.002, mu_s=5, g=0.9, n=1))
    tissue = ScatteringScene([cube, sphere])

    logger = EnergyLogger(tissue, defaultBinSize=0.1)
    source = PencilPointSource(position=Vector(0, 0, -29.99), direction=Vector(0, 0, 1), N=N, displaySize=1)

    source.propagate(tissue, logger=logger)

    viewer = Viewer(tissue, source, logger)
    viewer.reportStats()

    viewer.show2D(View2DProjectionX())
    viewer.show2D(View2DProjectionZ())
    viewer.show1D(Direction.Z_POS)
    viewer.show3D(visibility=Visibility.DEFAULT_3D | Visibility.VIEWS)


if __name__ == "__main__":
    exampleCode()
