from typing import List, Dict

from pytissueoptics.scene.geometry import Vector, SurfaceCollection
from pytissueoptics.scene.solids.stack.stackResult import StackResult


class CuboidStacker:
    """ Internal helper class to prepare and assemble a cuboid stack from 2 cuboids. """
    SURFACE_KEYS = ['left', 'right', 'bottom', 'top', 'front', 'back']
    SURFACE_PAIRS = [('left', 'right'), ('bottom', 'top'), ('front', 'back')]

    def __init__(self):
        self._onCuboid = None
        self._otherCuboid = None
        self._onSurfaceLabel = None
        self._otherSurfaceLabel = None
        self._newInterfaceIndex = None
        self._stackAxis = None

    def stack(self, onCuboid: 'Cuboid', otherCuboid: 'Cuboid', onSurface: str = 'top') -> StackResult:
        self._initStacking(onCuboid, otherCuboid, onSurface)
        self._translateOtherCuboid()
        self._configureInterfaceMaterial()
        return self._assemble()

    def _initStacking(self, onCuboid: 'Cuboid', otherCuboid: 'Cuboid', onSurfaceLabel: str):
        assert onSurfaceLabel in self.SURFACE_KEYS, f"Available surfaces to stack on are: {self.SURFACE_KEYS}"
        self._onCuboid = onCuboid
        self._otherCuboid = otherCuboid

        self._stackAxis = self._getSurfaceAxis(onSurfaceLabel)
        self._onSurfaceLabel = onSurfaceLabel
        self._otherSurfaceLabel = self._getOppositeSurface(onSurfaceLabel)
        onCuboidInterfaces = [label for label in self._onCuboid.surfaceLabels if "interface" in label]
        self._newInterfaceIndex = len(onCuboidInterfaces)

        self._validateShapeMatch()

    def _getSurfaceAxis(self, surfaceLabel: str) -> int:
        return max(axis if surfaceLabel in surfacePair else -1 for axis, surfacePair in enumerate(self.SURFACE_PAIRS))

    @property
    def _stackingTowardsPositive(self) -> bool:
        return self.SURFACE_PAIRS[self._stackAxis].index(self._onSurfaceLabel) == 1

    def _getOppositeSurface(self, surfaceLabel: str) -> str:
        onSurfaceIndex = self.SURFACE_PAIRS[self._stackAxis].index(surfaceLabel)
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
            oppositeEnvironment = self._otherCuboid.getEnvironment(self._otherSurfaceLabel)
        except:
            raise Exception("Ill-defined interface material: Can only stack another stack along its stacked axis.")
        self._onCuboid.setOutsideEnvironment(oppositeEnvironment, self._onSurfaceLabel)

        self._otherCuboid.setPolygons(surfaceLabel=self._otherSurfaceLabel,
                                      polygons=self._onCuboid.getPolygons(self._onSurfaceLabel))

    def _assemble(self) -> StackResult:
        return StackResult(shape=self._getStackShape(), position=self._getStackPosition(),
                           vertices=self._getStackVertices(), surfaces=self._getStackSurfaces(),
                           primitive=self._onCuboid.primitive, layerLabels=self._getLayerLabels())

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
        onCuboidVerticesIDs = {id(vertex) for vertex in self._onCuboid.vertices}
        newVertices = [vertex for vertex in self._otherCuboid.vertices if id(vertex) not in onCuboidVerticesIDs]
        stackVertices.extend(newVertices)
        return stackVertices

    def _getStackSurfaces(self) -> SurfaceCollection:
        surfaces = SurfaceCollection()
        surfaces.add(self._onSurfaceLabel, self._otherCuboid.getPolygons(self._onSurfaceLabel))
        surfaces.add(self._otherSurfaceLabel, self._onCuboid.getPolygons(self._otherSurfaceLabel))

        surfacesLeft = self.SURFACE_PAIRS[(self._stackAxis + 1) % 3] + self.SURFACE_PAIRS[(self._stackAxis + 2) % 3]
        for surfaceLabel in surfacesLeft:
            surfaces.add(surfaceLabel,
                         self._onCuboid.getPolygons(surfaceLabel) + self._otherCuboid.getPolygons(surfaceLabel))

        surfaces.extend(self._getStackInterfaces())
        return surfaces

    def _getStackInterfaces(self) -> SurfaceCollection:
        interfaces = SurfaceCollection()

        onCuboidInterfaces = [label for label in self._onCuboid.surfaceLabels if "interface" in label]
        for interface in onCuboidInterfaces:
            interfaces.add(interface, self._onCuboid.getPolygons(interface))

        newInterfaceLabel = f"interface{self._newInterfaceIndex}"
        interfaces.add(newInterfaceLabel, self._onCuboid.getPolygons(self._onSurfaceLabel))

        otherCuboidInterfaceLabels = [label for label in self._otherCuboid.surfaceLabels if "interface" in label]
        for i, otherInterfaceLabel in enumerate(otherCuboidInterfaceLabels):
            newOtherInterfaceLabel = f"interface{self._newInterfaceIndex + i + 1}"
            interfaces.add(newOtherInterfaceLabel, self._otherCuboid.getPolygons(otherInterfaceLabel))

        return interfaces

    def _getLayerLabels(self) -> Dict[str, List[str]]:
        self._correctInterfaceIndexesOfOtherCuboid()
        onCuboidLayerLabels = self._getUpdatedLayerLabelsOf(self._onCuboid)
        otherCuboidLayerLabels = self._getUpdatedLayerLabelsOf(self._otherCuboid)
        return dict(onCuboidLayerLabels, **otherCuboidLayerLabels)

    def _correctInterfaceIndexesOfOtherCuboid(self):
        if not self._otherCuboid.isStack():
            return

        for layerLabel, surfaceLabels in self._otherCuboid.getLayerLabelMap().items():
            for i, surfaceLabel in enumerate(surfaceLabels):
                if "interface" in surfaceLabel:
                    oldInterfaceIndex = int(surfaceLabel.split("interface")[1])
                    surfaceLabels[i] = f"interface{oldInterfaceIndex + self._newInterfaceIndex + 1}"

    def _getUpdatedLayerLabelsOf(self, cuboid: 'Cuboid') -> Dict[str, List[str]]:
        newInterfaceLabel = f"interface{self._newInterfaceIndex}"

        onCuboidLayerLabels = cuboid.getLayerLabelMap()
        if not onCuboidLayerLabels:
            onCuboidLayerLabels = {cuboid.getLabel(): cuboid.surfaceLabels}

        stackedSurfaceLabel = self._onSurfaceLabel if cuboid is self._onCuboid else self._otherSurfaceLabel
        for layerLabel, surfaceLabels in onCuboidLayerLabels.items():
            if stackedSurfaceLabel not in surfaceLabels:
                continue
            surfaceLabels.remove(stackedSurfaceLabel)
            surfaceLabels.append(newInterfaceLabel)

        return onCuboidLayerLabels
