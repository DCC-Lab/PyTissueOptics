import env
from pytissueoptics import *
import math

TITLE = "Hardware Accelerated Monte Carlo in Infinite Medium"

DESCRIPTION = """ Here we simulate the propagation of a divergent uniform source through an InfiniteTissue. We give the 
Tissue a material to propagate through and we use the useHardwareAcceleration flag to propagate the source. Since it is
parallelized, the simulation is done on all available cores. This allows to simulate in media that are more scattering
than when we use the series-based implementation. For now, hardware acceleration only supports InfiniteTissues. Usually,
it is not possible to view the data in 3D without binning as this generates too much data.
"""


def exampleCode():
    myMaterial = ScatteringMaterial(mu_s=30.0, mu_a=0.1, g=0.9)
    tissue = samples.InfiniteTissue(myMaterial)

    logger = EnergyLogger(tissue)
    source = DivergentSource(position=Vector(0, 0, 0), direction=Vector(0, 0, 1), N=10000,
                             diameter=0.2, divergence=math.pi/4, useHardwareAcceleration=True)

    source.propagate(tissue, logger=logger)

    viewer = Viewer(tissue, source, logger)
    viewer.reportStats()

    viewer.show2D(View2DProjectionX(limits=((0, 2), (-1, 1))))


if __name__ == "__main__":
    exampleCode()
