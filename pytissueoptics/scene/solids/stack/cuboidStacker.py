from typing import List
from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids.stack.stackResult import StackResult
from pytissueoptics.scene.geometry import SurfaceCollection


class CuboidStacker:
    """ Internal helper class to prepare and assemble a cuboid stack from 2 cuboids. """
    SURFACE_KEYS = ['Left', 'Right', 'Bottom', 'Top', 'Front', 'Back']
    SURFACE_PAIRS = [('Left', 'Right'), ('Bottom', 'Top'), ('Front', 'Back')]

    def __init__(self):
        self._onCuboid = None
        self._otherCuboid = None
        self._onSurfaceName = None
        self._otherSurfaceName = None
        self._stackAxis = None

    def stack(self, onCuboid: 'Cuboid', otherCuboid: 'Cuboid', onSurface: str = 'Top') -> StackResult:
        self._initStacking(onCuboid, otherCuboid, onSurface)
        self._translateOtherCuboid()
        self._configureInterfaceMaterial()
        return self._assemble()

    def _initStacking(self, onCuboid: 'Cuboid', otherCuboid: 'Cuboid', onSurfaceName: str):
        assert onSurfaceName in self.SURFACE_KEYS, f"Available surfaces to stack on are: {self.SURFACE_KEYS}"
        self._onCuboid = onCuboid
        self._otherCuboid = otherCuboid

        self._stackAxis = self._getSurfaceAxis(onSurfaceName)
        self._onSurfaceName = onSurfaceName
        self._otherSurfaceName = self._getOppositeSurface(onSurfaceName)

        self._validateShapeMatch()

    def _getSurfaceAxis(self, surfaceName: str) -> int:
        return max(axis if surfaceName in surfacePair else -1 for axis, surfacePair in enumerate(self.SURFACE_PAIRS))

    @property
    def _stackingTowardsPositive(self) -> bool:
        return self.SURFACE_PAIRS[self._stackAxis].index(self._onSurfaceName) == 1

    def _getOppositeSurface(self, surfaceName: str) -> str:
        onSurfaceIndex = self.SURFACE_PAIRS[self._stackAxis].index(surfaceName)
        return self.SURFACE_PAIRS[self._stackAxis][(onSurfaceIndex + 1) % 2]

    def _validateShapeMatch(self):
        fixedAxes = [0, 1, 2]
        fixedAxes.remove(self._stackAxis)
        for fixedAxis in fixedAxes:
            assert self._onCuboid.shape[fixedAxis] == self._otherCuboid.shape[fixedAxis], \
                f"Stacking of mismatched surfaces is not supported."

    def _translateOtherCuboid(self):
        relativePosition = [0, 0, 0]
        relativePosition[self._stackAxis] = self._onCuboid.shape[self._stackAxis] / 2 + \
                                            self._otherCuboid.shape[self._stackAxis] / 2
        relativePosition = Vector(*relativePosition)

        if not self._stackingTowardsPositive:
            relativePosition.multiply(-1)

        self._otherCuboid.translateTo(self._onCuboid.position + relativePosition)

    def _configureInterfaceMaterial(self):
        """ Set new interface material and remove duplicate surfaces. """
        try:
            oppositeMaterial = self._otherCuboid.getMaterial(self._otherSurfaceName)
        except:
            raise Exception("Ill-defined interface material: Can only stack another stack along its stacked axis.")
        self._onCuboid.setOutsideMaterial(oppositeMaterial, self._onSurfaceName)

        self._otherCuboid.setPolygons(surfaceName=self._otherSurfaceName,
                                      polygons=self._onCuboid.getPolygons(self._onSurfaceName))

    def _assemble(self) -> StackResult:
        return StackResult(shape=self._getStackShape(), position=self._getStackPosition(),
                           vertices=self._getStackVertices(), surfaces=self._getStackSurfaces(),
                           primitive=self._onCuboid.primitive)

    def _getStackShape(self) -> List[float]:
        stackShape = self._onCuboid.shape.copy()
        stackShape[self._stackAxis] += self._otherCuboid.shape[self._stackAxis]
        return stackShape

    def _getStackPosition(self):
        relativeStackCentroid = [0, 0, 0]
        relativeStackCentroid[self._stackAxis] = self._otherCuboid.shape[self._stackAxis] / 2
        relativeStackCentroid = Vector(*relativeStackCentroid)
        if not self._stackingTowardsPositive:
            relativeStackCentroid.multiply(-1)
        return self._onCuboid.position + relativeStackCentroid

    def _getStackVertices(self):
        stackVertices = self._onCuboid.vertices
        newVertices = [vertex for vertex in self._otherCuboid.vertices if vertex not in self._onCuboid.vertices]
        stackVertices.extend(newVertices)
        return stackVertices

    def _getStackSurfaces(self) -> SurfaceCollection:
        surfaces = SurfaceCollection()
        surfaces.add(self._onSurfaceName, self._otherCuboid.getPolygons(self._onSurfaceName))
        surfaces.add(self._otherSurfaceName, self._onCuboid.getPolygons(self._otherSurfaceName))

        surfacesLeft = self.SURFACE_PAIRS[(self._stackAxis + 1) % 3] + self.SURFACE_PAIRS[(self._stackAxis + 2) % 3]
        for surfaceName in surfacesLeft:
            surfaces.add(surfaceName,
                         self._onCuboid.getPolygons(surfaceName) + self._otherCuboid.getPolygons(surfaceName))

        surfaces.extend(self._getStackInterfaces())
        return surfaces

    def _getStackInterfaces(self) -> SurfaceCollection:
        interfaces = SurfaceCollection()

        onCuboidInterfaces = [name for name in self._onCuboid.surfaceNames if "Interface" in name]
        for interface in onCuboidInterfaces:
            interfaces.add(interface, self._onCuboid.getPolygons(interface))

        newInterfaceIndex = len(onCuboidInterfaces)
        interfaces.add(f'Interface{newInterfaceIndex}', self._onCuboid.getPolygons(self._onSurfaceName))

        otherCuboidInterfaceKeys = [name for name in self._otherCuboid.surfaceNames if "Interface" in name]
        for i, otherInterfaceKey in enumerate(otherCuboidInterfaceKeys):
            newOtherInterfaceIndex = newInterfaceIndex + 1 + i
            interfaces.add(f'Interface{newOtherInterfaceIndex}', self._otherCuboid.getPolygons(otherInterfaceKey))

        return interfaces
