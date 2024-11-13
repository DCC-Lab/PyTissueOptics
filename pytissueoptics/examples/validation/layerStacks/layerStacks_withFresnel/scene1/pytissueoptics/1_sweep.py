
from pytissueoptics import *


class ValidationStack(ScatteringScene):
    """ Phantom tissue consisting of 2 layers with various optical properties. """
    TISSUE = []

    def __init__(self, n1, n2, worldMaterial=ScatteringMaterial()):
        self._create(n1, n2)
        super().__init__(self.TISSUE, worldMaterial)

    def _create(self, n1, n2):
        mu_s = [20, 0]
        mu_a = [0.1, 10]
        g = [0.7, 1]
        w = 50
        t = [1, 10]
        firstLayer = Cuboid(w, w, t[0], material=ScatteringMaterial(mu_s[0], mu_a[0], g[0], n1), label="firstLayer")
        secondLayer = Cuboid(w, w, t[1], material=ScatteringMaterial(mu_s[1], mu_a[1], g[1], n2), label="secondLayer")
        layerStack = secondLayer.stack(firstLayer, 'front')
        layerStack.translateTo(Vector(0, 0, sum(t) / 2))
        self.TISSUE = [layerStack]


def exampleCode():
    N = 1000000 if hardwareAccelerationIsAvailable() else 1000

    n1n2 = [(1, 1.0), (1, 1.1), (1, 1.2), (1, 1.3), (1, 1.4), (1, 1.5), (1, 1.6), (1, 1.7), (1, 1.8), (1, 1.9), (1, 2)]
    results = []
    for n1, n2 in n1n2:
        layerStack = ValidationStack(n1, n2)
        logger = EnergyLogger(layerStack)
        source = PencilPointSource(position=Vector(0, 0, -0.01), direction=Vector(0, 0, 1), N=N)
        source.propagate(layerStack, logger=logger)
        viewer = Viewer(layerStack, source, logger)
        viewer.reportStats("firstLayer")
        viewer.reportStats("secondLayer")
        stats = Stats(logger)

        results.append((stats.getAbsorbance("firstLayer", useTotalEnergy=True), stats.getAbsorbance("secondLayer", useTotalEnergy=True)))
    print(results)

if __name__ == "__main__":
    exampleCode()
