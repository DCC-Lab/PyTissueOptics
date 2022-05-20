from pytissueoptics import *

TITLE = "PencilSource propagation through 3 layers of tissue"

DESCRIPTION = """ Here we simulate the propagation of a pencil source through a custom tissue called PhantomTissue.
This tissue is composed of a stacked cuboid with 3 layers of different material.
Then we propagate the PencilSource photons in the tissue and then show the distribution of the energy in the tissue at various locations.
"""


def exampleCode():
    logger = Logger()
    tissue = tissues.PhantomTissue()
    source = PencilPointSource(position=Vector(0, 0, -1), direction=Vector(0, 0, 1), N=2000)

    source.propagate(tissue, logger=logger)

    stats = Stats(logger, source, tissue)
    stats.report()

    displayConfig = DisplayConfig(showPointsAsSpheres=False)
    stats.showEnergy3D(config=displayConfig)
    stats.showEnergy3DOfSurfaces()
    stats.showEnergy2D()
    stats.showEnergy2D("middleLayer", bins=51)
    stats.showEnergy2D("middleLayer", "interface1", projection='z', bins=51, logScale=True, enteringSurface=True)
    stats.showEnergy1D(bins=100)


if __name__ == "__main__":
    exampleCode()
