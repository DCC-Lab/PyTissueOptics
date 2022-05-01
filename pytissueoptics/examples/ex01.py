TITLE = "PencilSource propagation through 3 layers of tissue"
DESCRIPTION = """ Here we simulate the propagation of a pencil source through a custom tissue called PhantomTissue.
This tissue is composed of a stacked cuboid with 3 layers with each a different material.
Then we propagate the PencilSource photons in the tissue and then show the distribution of the energy in the tissue on multiple locations.
"""


def exampleCode():
    from pytissueoptics.rayscattering import PencilSource, Stats
    from pytissueoptics.rayscattering.tissues import PhantomTissue
    from pytissueoptics.scene import Vector, Logger

    logger = Logger()
    tissue = PhantomTissue()
    source = PencilSource(position=Vector(0, 0, -1), direction=Vector(0, 0, 1), N=2000)

    source.propagate(tissue, logger=logger)

    stats = Stats(logger, source, tissue)
    stats.report()
    stats.showEnergy3D()
    stats.showEnergy3DOfSurfaces()
    stats.showEnergy2D()
    stats.showEnergy2D("middleLayer", bins=51)
    stats.showEnergy2D("middleLayer", "interface1", projection='z', bins=51, logScale=True, enteringSurface=True)
    stats.showEnergy1D(bins=100)


if __name__ == "__main__":
    exampleCode()
