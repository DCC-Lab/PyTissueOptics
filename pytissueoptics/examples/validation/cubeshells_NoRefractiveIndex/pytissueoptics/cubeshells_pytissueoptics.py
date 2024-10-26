from pytissueoptics import *

TITLE = "MCX Spherical shells"

DESCRIPTION = """ Adapted MCX built-in example - a 4-layer heterogeneous domain including a thin spherical-shell with 
low-scattering/low absorption material to simulate CSF-like tissue in the brain. """


def exampleCode():
    N = 1000000 if hardwareAccelerationIsAvailable() else 100

    outerShell = Cube(13, material=ScatteringMaterial(mu_a=0.01, mu_s=2, g=0.7, n=1), label="outer")
    innerShell = Cube(10,  material=ScatteringMaterial(mu_a=1, mu_s=0.2, g=0.9, n=1), label="inner")
    core = Cube(5, material=ScatteringMaterial(mu_a=0.2, mu_s=20, g=0.95, n=1), label="core")

    tissue = ScatteringScene([core, innerShell, outerShell])
    logger = EnergyLogger(tissue, defaultBinSize=0.1)
    source = PencilPointSource(position=Vector(0, 0, -30), direction=Vector(0, 0, 1), N=N, displaySize=2)

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
