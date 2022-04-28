from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.rayscattering.opencl.CLPhotons import CLPhotons
from pytissueoptics.scene import Vector, Logger
from pytissueoptics.scene.geometry import Environment


class CLSource:
    def __init__(self, position: Vector, direction: Vector, N: int, worldMaterial: ScatteringMaterial):
        self.position = position
        self.direction = direction
        self.direction.normalize()
        self.N = N
        self.worldMaterial = worldMaterial
        self._environment = None

    def propagate(self, worldMaterial: ScatteringMaterial, logger: Logger = None):
        photons = CLPhotons(self, worldMaterial, logger)
        photons.propagate()

    def getPhotonCount(self):
        return self.N

    def getEnvironment(self) -> Environment:
        if self._environment is None:
            return Environment(None)
        return self._environment


class CLPencilSource(CLSource):
    def __init__(self, position=Vector(0, 0, 0), direction=Vector(0, 0, 1), N=100, worldMaterial=ScatteringMaterial()):
        super().__init__(position, direction, N, worldMaterial)


class CLIsotropicSource(CLSource):
    def __init__(self, position=Vector(0, 0, 0), direction=Vector(0, 0, 1), N=100, worldMaterial=ScatteringMaterial()):
        super().__init__(position, direction, N, worldMaterial)
