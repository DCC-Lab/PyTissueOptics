from pytissueoptics import *

TITLE = "PencilSource propagation through a multi-layered tissue"

DESCRIPTION = """ Propagation of a pencil source through a sample tissue called PhantomTissue. This tissue is composed 
of a stacked cuboid made of 3 layers of different material. """


def exampleCode():
    N = 100000 if hardwareAccelerationIsAvailable() else 1000

    tissue = samples.PhantomTissue()
    logger = EnergyLogger(tissue)
    source = PencilPointSource(position=Vector(0, 0, -1), direction=Vector(0, 0, 1), N=N)

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
    import env
    exampleCode()
