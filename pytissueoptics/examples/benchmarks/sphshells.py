import env  # noqa: F401
from pytissueoptics import *  # noqa: F403

TITLE = "MCX Spherical shells"

DESCRIPTION = """ Adapted MCX built-in example - a 4-layer heterogeneous domain including a thin spherical-shell with 
low-scattering/low absorption material to simulate CSF-like tissue in the brain. """


def exampleCode():
    N = 100000 if hardwareAccelerationIsAvailable() else 100

    outerShell = Sphere(2.5, order=2, material=ScatteringMaterial(mu_a=0.04, mu_s=0.09, g=0.89, n=1.37), label="outer")
    innerShell = Sphere(2.3, order=2, material=ScatteringMaterial(mu_a=0.2, mu_s=90, g=0.89, n=1.37), label="inner")
    core = Sphere(1.0, order=2, material=ScatteringMaterial(mu_a=0.5, mu_s=1e-6, g=1, n=1.37), label="core")
    grid = Cuboid(6, 6, 6, material=ScatteringMaterial(mu_a=0.2, mu_s=70, g=0.89, n=1.37))
    tissue = ScatteringScene([core, innerShell, outerShell, grid])
    logger = EnergyLogger(tissue, defaultBinSize=0.1)
    source = PencilPointSource(position=Vector(0, 0.01, -3), direction=Vector(0, 0, 1), N=N, displaySize=2)

    source.propagate(tissue, logger=logger)

    viewer = Viewer(tissue, source, logger)
    viewer.reportStats()

    viewer.show2D(View2DProjectionX())
    viewer.show2D(View2DProjectionZ())
    viewer.show2D(View2DProjectionX(solidLabel="outer"))
    viewer.show2D(View2DProjectionX(solidLabel="inner"))
    viewer.show2D(View2DProjectionX(solidLabel="core"))

    viewer.show2D(View2DSurfaceZ(solidLabel="outer", surfaceLabel="ellipsoid"))
    viewer.show1D(Direction.Z_POS)
    viewer.show3D(pointCloudStyle=PointCloudStyle(showSolidPoints=False))
    viewer.show3D()


if __name__ == "__main__":
    exampleCode()
