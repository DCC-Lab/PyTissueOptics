from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids.stackResult import StackResult


class CuboidStacker:
    """ Internal helper class to prepare and assemble a cuboid stack from 2 cuboids. """
    def __init__(self):
        self._surfacePairs = [('Left', 'Right'), ('Bottom', 'Top'), ('Front', 'Back')]
        self._surfaceKeys = ['Left', 'Right', 'Bottom', 'Top', 'Front', 'Back']

        self._onCuboid = None
        self._otherCuboid = None
        self._onSurfaceKey = None
        self._otherSurfaceKey = None
        self._stackAxis = None

    def stack(self, onCuboid: 'Cuboid', otherCuboid: 'Cuboid', onSurface: str = 'Top') -> StackResult:
        self._initStacking(onCuboid, otherCuboid, onSurface)
        self._translateOtherCuboid()
        self._configureInterfaceMaterial()
        return self._assemble()

    def _initStacking(self, onCuboid, otherCuboid, onSurface):
        assert onSurface in self._surfaceKeys, f"Available surfaces to stack on are: {self._surfaceKeys}"
        self._onCuboid = onCuboid
        self._otherCuboid = otherCuboid
        self._onSurfaceKey = onSurface

        self._stackAxis = self._axisOfSurface(onSurface)
        onSurfaceIndex = self._surfacePairs[self._stackAxis].index(onSurface)
        self._otherSurfaceKey = self._surfacePairs[self._stackAxis][(onSurfaceIndex + 1) % 2]

        self._validateShapeMatch()

    def _axisOfSurface(self, surfaceKey) -> int:
        return max(axis if surfaceKey in surfacePair else -1 for axis, surfacePair in enumerate(self._surfacePairs))

    def _validateShapeMatch(self):
        fixedAxes = [0, 1, 2]
        fixedAxes.remove(self._stackAxis)
        for fixedAxis in fixedAxes:
            assert self._onCuboid.shape[fixedAxis] == self._otherCuboid.shape[fixedAxis], \
               f"Stacking of mismatched surfaces is not supported."

    def _translateOtherCuboid(self):
        relativePosition = [0, 0, 0]
        relativePosition[self._stackAxis] = self._onCuboid.shape[self._stackAxis]/2 + self._otherCuboid.shape[self._stackAxis]/2
        relativePosition = Vector(*relativePosition)
        self._otherCuboid.translateTo(self._onCuboid.position + relativePosition)

    def _configureInterfaceMaterial(self):
        """ Set new interface material and remove duplicate surfaces. """
        oppositeMaterial = self._otherCuboid._surfaceDict[self._otherSurfaceKey][0].insideMaterial
        for oppositeSurface in self._otherCuboid._surfaceDict[self._otherSurfaceKey]:
            assert oppositeSurface.insideMaterial == oppositeMaterial, \
                "Ill-defined interface material: Cannot stack another stack along it's stacked axis."

        self._onCuboid._setOutsideMaterial(oppositeMaterial, faceKey=self._onSurfaceKey)
        self._otherCuboid._surfaceDict[self._otherSurfaceKey] = self._onCuboid._surfaceDict[self._onSurfaceKey]

    def _assemble(self) -> StackResult:
        return StackResult(shape=self._getStackShape(), position=self._getStackPosition(),
                           vertices=self._getStackVertices(), surfaces=self._getStackSurfaces(),
                           interfaces=self._getStackInterfaces(), primitive=self._onCuboid._primitive)

    def _getStackShape(self):
        stackShape = self._onCuboid.shape.copy()
        stackShape[self._stackAxis] += self._otherCuboid.shape[self._stackAxis]
        return stackShape

    def _getStackPosition(self):
        relativeStackCentroid = [0, 0, 0]
        relativeStackCentroid[self._stackAxis] = self._otherCuboid.shape[self._stackAxis] / 2
        return self._onCuboid.position + Vector(*relativeStackCentroid)

    def _getStackVertices(self):
        stackVertices = self._onCuboid._vertices
        newVertices = [vertex for vertex in self._otherCuboid._vertices if vertex not in self._onCuboid._vertices]
        stackVertices.extend(newVertices)
        return stackVertices

    def _getStackSurfaces(self):
        stackSurfaces = {self._onSurfaceKey: self._otherCuboid._surfaceDict[self._onSurfaceKey],
                         self._otherSurfaceKey: self._onCuboid._surfaceDict[self._otherSurfaceKey]}

        surfaceKeysLeft = self._surfacePairs[(self._stackAxis + 1) % 3] + self._surfacePairs[(self._stackAxis + 2) % 3]
        for surfaceKey in surfaceKeysLeft:
            stackSurfaces[surfaceKey] = self._onCuboid._surfaceDict[surfaceKey] + self._otherCuboid._surfaceDict[surfaceKey]
        return stackSurfaces

    def _getStackInterfaces(self):
        stackInterfaces = {}

        onCuboidInterfaceKeys = [key for key in self._onCuboid._surfaceDict.keys() if "Interface" in key]
        for interfaceKey in onCuboidInterfaceKeys:
            stackInterfaces[interfaceKey] = self._onCuboid._surfaceDict[interfaceKey]

        newInterfaceIndex = len(onCuboidInterfaceKeys)
        stackInterfaces[f'Interface{newInterfaceIndex}'] = self._onCuboid._surfaceDict[self._onSurfaceKey]

        otherCuboidInterfaceKeys = [key for key in self._otherCuboid._surfaceDict.keys() if "Interface" in key]
        for i, otherInterfaceKey in enumerate(otherCuboidInterfaceKeys):
            newOtherInterfaceIndex = newInterfaceIndex + 1 + i
            stackInterfaces[f'Interface{newOtherInterfaceIndex}'] = self._otherCuboid._surfaceDict[otherInterfaceKey]

        return stackInterfaces
