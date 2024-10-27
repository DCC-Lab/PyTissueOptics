from pytissueoptics import *

TITLE = "MCX Spherical shells"

DESCRIPTION = """ Adapted MCX built-in example - a 4-layer heterogeneous domain including a thin spherical-shell with 
low-scattering/low absorption material to simulate CSF-like tissue in the brain. """


def exampleCode():
    N = 1000000 if hardwareAccelerationIsAvailable() else 100
    n = [1, 1, 1]
    mu_s = [0.1, 2, 1]
    mu_a = [0.01, 0.2, 0.01]
    g = [0.99, 0.9, 0.7]
    W = 13
    medium = Cube(edge=W, material=ScatteringMaterial(mu_s[0], mu_a[0], g[0], n[0]), label="Medium")
    shell = Cube(edge=10, material=ScatteringMaterial(mu_s[1], mu_a[1], g[1], n[1]), label="Shell")
    core = Cube(edge=9, material=ScatteringMaterial(mu_s[2], mu_a[2], g[2], n[2]), label="Core")

    tissue = ScatteringScene([medium, shell, core])
    logger = EnergyLogger(tissue, defaultBinSize=0.1)
    source = PencilPointSource(position=Vector(0, 0, -13), direction=Vector(0, 0, 1), N=N, displaySize=2)

    source.propagate(tissue, logger=logger)

    viewer = Viewer(tissue, source, logger)
    viewer.reportStats()
    viewer.show2D(View2DProjectionX())


if __name__ == "__main__":
    exampleCode()
