import env  # noqa: F401
from pytissueoptics import *  # noqa: F403

TITLE = "Divergent source propagation through a multi-layered tissue"

DESCRIPTION = """ Propagation of a divergent source through a sample tissue called PhantomTissue. This tissue is composed 
of a stacked cuboid made of 3 layers of different material. """


def exampleCode():
    N = 100000 if hardwareAccelerationIsAvailable() else 1000

    tissue = samples.PhantomTissue()
    logger = EnergyLogger(tissue)
    source = DivergentSource(position=Vector(0, 0, -0.2), direction=Vector(0, 0, 1), N=N,
                             diameter=0.1, divergence=0.4, displaySize=0.2)

    tissue.show(source=source)

    source.propagate(tissue, logger=logger)

    viewer = Viewer(tissue, source, logger)
    viewer.reportStats()

    viewer.show2D(View2DProjectionX())
    viewer.show2D(View2DProjectionX(solidLabel="middleLayer"))
    viewer.show2D(View2DSurfaceZ(solidLabel="middleLayer", surfaceLabel="interface1", surfaceEnergyLeaving=False))
    viewer.show1D(Direction.Z_POS)
    viewer.show3D()
    viewer.show3D(pointCloudStyle=PointCloudStyle(showSolidPoints=False))


if __name__ == "__main__":
    exampleCode()
