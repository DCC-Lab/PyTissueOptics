from pytissueoptics import *

TITLE = "MCX Spherical shells"

DESCRIPTION = """ Adapted MCX built-in example - a 4-layer heterogeneous domain including a thin spherical-shell with 
low-scattering/low absorption material to simulate CSF-like tissue in the brain. """


def exampleCode():
    import numpy as np
    np.random.seed(651)
    N = 10000 if hardwareAccelerationIsAvailable() else 1000

    outerShell = Sphere(25, order=3, material=ScatteringMaterial(mu_a=0.004, mu_s=0.009, g=0.89, n=1.37))
    innerShell = Sphere(23, order=3, material=ScatteringMaterial(mu_a=0.02, mu_s=9, g=0.89, n=1.37))
    core = Sphere(10, order=3, material=ScatteringMaterial(mu_a=0.05, mu_s=0, g=1, n=1.37))
    tissue = ScatteringScene([outerShell, innerShell, core])

    logger = EnergyLogger(tissue)
    source = PencilPointSource(position=Vector(0, 0, -30), direction=Vector(0, 0, 1), N=N)

    source.propagate(tissue, logger=logger)

    viewer = Viewer(tissue, source, logger)
    viewer.reportStats()

    viewer.show2D(View2DProjectionX())
    viewer.show1D(Direction.Z_POS)


if __name__ == "__main__":
    import env
    exampleCode()
