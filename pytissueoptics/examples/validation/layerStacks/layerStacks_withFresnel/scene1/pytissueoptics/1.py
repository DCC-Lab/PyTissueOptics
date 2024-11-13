
from pytissueoptics import *    

TITLE = "Divergent source propagation through a multi-layered tissue"

DESCRIPTION = """ Propagation of a divergent source through a sample tissue called PhantomTissue. This tissue is composed 
of a stacked cuboid made of 3 layers of different material. """


class ValidationStack(ScatteringScene):
    """ Phantom tissue consisting of 3 layers with various optical properties. """
    TISSUE = []

    def __init__(self, worldMaterial=ScatteringMaterial()):
        self._create()
        super().__init__(self.TISSUE, worldMaterial)

    def _create(self):
        n = [1.3, 1.4, 1.5]
        mu_s = [20, 2, 200]
        mu_a = [0.1, 10, 2]
        g = [0.7, 0.9, 0.95]
        w = 50
        t = [1, 0.5, 1]

        frontLayer = Cuboid(w, w, t[0], material=ScatteringMaterial(mu_s[0], mu_a[0], g[0], n[0]), label="frontLayer")
        middleLayer = Cuboid(w, w, t[1], material=ScatteringMaterial(mu_s[1], mu_a[1], g[1], n[1]), label="middleLayer")
        backLayer = Cuboid(w, w, t[2], material=ScatteringMaterial(mu_s[2], mu_a[2], g[2], n[2]), label="backLayer")
        layerStack = backLayer.stack(middleLayer, 'front').stack(frontLayer, 'front')
        layerStack.translateTo(Vector(0, 0, sum(t) / 2))

        self.TISSUE = [layerStack]


def exampleCode():
    N = 10000 if hardwareAccelerationIsAvailable() else 10000
    layerStack = ValidationStack()
    logger = EnergyLogger(layerStack)
    source = PencilPointSource(position=Vector(0, 0, -0.01), direction=Vector(0, 0, 1), N=N, useHardwareAcceleration=False)

    #layerStack.show(source=source)

    source.propagate(layerStack, logger=logger)

    viewer = Viewer(layerStack, source, logger)
    viewer.reportStats()

    #viewer.show2D(View2DProjectionX())
    #viewer.show2D(View2DProjectionX(solidLabel="middleLayer"))
    #viewer.show2D(View2DSurfaceZ(solidLabel="middleLayer", surfaceLabel="interface1", surfaceEnergyLeaving=False))
    #viewer.show1D(Direction.Z_POS)
    #viewer.show3D()
    #viewer.show3D(pointCloudStyle=PointCloudStyle(showSolidPoints=False))


if __name__ == "__main__":
    exampleCode()
