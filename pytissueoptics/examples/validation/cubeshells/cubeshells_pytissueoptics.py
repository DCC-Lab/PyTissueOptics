from pytissueoptics import *

TITLE = "MCX Spherical shells"

DESCRIPTION = """ Adapted MCX built-in example - a 4-layer heterogeneous domain including a thin spherical-shell with 
low-scattering/low absorption material to simulate CSF-like tissue in the brain. """


def exampleCode():
    N = 1000000 if hardwareAccelerationIsAvailable() else 100

    medium = Cube(edge=60, position=Vector(0, 0, 0), material=ScatteringMaterial(mu_a=0.02, mu_s=7, g=0.89, n=1.37), label="grid")
    outerShell = Cube(50, material=ScatteringMaterial(mu_a=0.004, mu_s=0.009, g=0.89, n=1.37), label="outer")
    innerShell = Cube(46,  material=ScatteringMaterial(mu_a=0.02, mu_s=9, g=0.89, n=1.37), label="inner")
    core = Cube(20, material=ScatteringMaterial(mu_a=0.05, mu_s=1E-10, g=1, n=1.37), label="core")

    tissue = ScatteringScene([core, innerShell, outerShell, medium])
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
