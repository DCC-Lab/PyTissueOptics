from pytissueoptics import *

TITLE = "Hardware Accelerated Monte Carlo in Infinite Medium"
DESCRIPTION = """ Here we simulate the propagation of a pencil source through an InfiniteTissue. We give the Tissue a
material to propagate through and we use the useHardwareAcceleration flag to propagate the source. Since it is
parallelized, the simulation is done on all available cores. This allows to simulate in media that are more scattering
then when we use the series-based implementation. For now, hardware acceleration only supports InfiniteTissues. Usually,
it is not possible to view the data in 3D without binning as this generates too much data.
"""


def exampleCode():
    logger = Logger()
    myMaterial = ScatteringMaterial(mu_s=30.0, mu_a=0.1, g=0.9)
    tissue = tissues.InfiniteTissue(myMaterial)
    source = PencilSource(position=Vector(0, 0, 0), direction=Vector(0, 0, 1), N=50000, useHardwareAcceleration=True)

    source.propagate(tissue, logger=logger)

    stats = Stats(logger, source, tissue)
    stats.report()

    stats.showEnergy2D(projection='y', bins=101, limits=[[-1, 1], [-0.1, 1]])


if __name__ == "__main__":
    exampleCode()
