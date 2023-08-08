import exampleenv
from pytissueoptics import *

TITLE = "PencilSource propagation through 3 layers of tissue"

DESCRIPTION = """ Here we simulate the propagation of a pencil source through a custom tissue called PhantomTissue.
This tissue is composed of a stacked cuboid with 3 layers of different material.
Then we propagate the PencilSource photons in the tissue and then show the distribution of the energy in the tissue at various locations.
"""


def exampleCode():
    tissue = samples.PhantomTissue()
    logger = EnergyLogger(tissue)
    source = PencilPointSource(position=Vector(0, 0, -1), direction=Vector(0, 0, 1), N=20000)

    source.propagate(tissue, logger=logger)

    viewer = Viewer(tissue, source, logger)
    viewer.reportStats()

    viewer.show3D()
    viewer.show3D(pointCloudStyle=PointCloudStyle(showSolidPoints=False))
    viewer.show2D(View2DProjectionX())
    viewer.show2D(View2DProjectionX(solidLabel="middleLayer"))
    viewer.show2D(View2DSurfaceZ(solidLabel="middleLayer", surfaceLabel="interface1", surfaceEnergyLeaving=False))
    viewer.show1D(Direction.Z_POS)


if __name__ == "__main__":
    exampleCode()
