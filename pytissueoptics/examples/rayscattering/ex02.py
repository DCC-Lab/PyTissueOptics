from pytissueoptics import *

TITLE = "Propagation in an Infinite Medium"

DESCRIPTION = """ Propagation of a divergent uniform source through an InfiniteTissue. The optical properties of the
tissue are set to mimic a typical biological tissue. It is not recommended to try to visualize the data in 3D without 
binning as this generates a lot of data. """


def exampleCode():
    import math
    N = 10000 if hardwareAccelerationIsAvailable() else 25

    myMaterial = ScatteringMaterial(mu_s=30.0, mu_a=0.1, g=0.9)
    tissue = samples.InfiniteTissue(myMaterial)

    logger = EnergyLogger(tissue)
    source = DivergentSource(position=Vector(0, 0, 0), direction=Vector(0, 0, 1), N=N,
                             diameter=0.2, divergence=math.pi/4)

    source.propagate(tissue, logger=logger)

    viewer = Viewer(tissue, source, logger)
    viewer.reportStats()
    viewer.show2D(View2DProjectionX())
    viewer.show2D(View2DProjectionX(limits=((0, 2), (-1, 1))))


if __name__ == "__main__":
    import env
    exampleCode()
