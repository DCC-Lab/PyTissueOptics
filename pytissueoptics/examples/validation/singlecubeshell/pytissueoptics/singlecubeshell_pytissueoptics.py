from pytissueoptics import *

TITLE = "MCX Spherical shells"

DESCRIPTION = """ Adapted MCX built-in example - a 4-layer heterogeneous domain including a thin spherical-shell with 
low-scattering/low absorption material to simulate CSF-like tissue in the brain. """


def exampleCode():
    N = 1000000 if hardwareAccelerationIsAvailable() else 100

    medium = Cube(edge=60, position=Vector(0, 0, 0), material=ScatteringMaterial(mu_a=0.02, mu_s=7, g=0.89, n=1.37), label="grid")
    outerShell = Cube(50, material=ScatteringMaterial(mu_a=0.004, mu_s=0.009, g=0.89, n=1.37), label="outer")

    tissue = ScatteringScene([outerShell, medium])
    logger = EnergyLogger(tissue, defaultBinSize=0.1)
    source = PencilPointSource(position=Vector(0, 0, -30), direction=Vector(0, 0, 1), N=N, displaySize=2)

    source.propagate(tissue, logger=logger)

    viewer = Viewer(tissue, source, logger)
    viewer.reportStats()

    viewer.show2D(View2DProjectionX())
    viewer.show2D(View2DProjectionZ())
    viewer.show3D()


if __name__ == "__main__":
    exampleCode()
