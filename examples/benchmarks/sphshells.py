from pytissueoptics import *

TITLE = "MCX Spherical shells"

DESCRIPTION = """ Adapted MCX built-in example - a 4-layer heterogeneous domain including a thin spherical-shell with 
low-scattering/low absorption material to simulate CSF-like tissue in the brain. """


def exampleCode():
    import numpy as np
    np.random.seed(651)
    N = 1000 if hardwareAccelerationIsAvailable() else 100

    outerShell = Sphere(30, order=4, material=ScatteringMaterial(mu_a=0.004, mu_s=0.009, g=0.89, n=1.37), label="outer")
    innerShell = Sphere(23, order=4, material=ScatteringMaterial(mu_a=0.02, mu_s=9, g=0.89, n=1.37), label="inner")
    core = Sphere(10, order=4, material=ScatteringMaterial(mu_a=0.05, mu_s=0.00001, g=1, n=1.37), label="core")

    tissue = ScatteringScene([core, innerShell, outerShell], ignoreIntersections=True)
    outerShell.setOutsideEnvironment(tissue.getWorldEnvironment())
    innerShell.setOutsideEnvironment(outerShell.getEnvironment())
    core.setOutsideEnvironment(innerShell.getEnvironment())

    logger = EnergyLogger(tissue, defaultBinSize=0.1)
    source = DivergentSource(position=Vector(0, 0, -30), direction=Vector(0, 0, 1), N=N,
                             useHardwareAcceleration=True, diameter=2, divergence=0.1)

    source.propagate(tissue, logger=logger)

    viewer = Viewer(tissue, source, logger)
    viewer.reportStats()

    viewer.show2D(View2DProjectionX())
    viewer.show2D(View2DProjectionZ())
    viewer.show2D(View2DSurfaceZ(solidLabel="outer", surfaceLabel="ellipsoid"))
    viewer.show1D(Direction.Z_POS)
    viewer.show3D()


if __name__ == "__main__":
    import env
    exampleCode()
