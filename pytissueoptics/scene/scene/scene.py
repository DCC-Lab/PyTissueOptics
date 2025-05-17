import sys
import warnings
from typing import Dict, List, Optional

from pytissueoptics.scene.geometry import INTERFACE_KEY, BoundingBox, Environment, Polygon, Vector
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.viewer import Abstract3DViewer, Displayable


class Scene(Displayable):
    def __init__(self, solids: List[Solid] = None, ignoreIntersections=False, worldMaterial=None):
        self._solids: List[Solid] = []
        self._ignoreIntersections = ignoreIntersections
        self._solidsContainedIn: Dict[str, List[str]] = {}
        self._worldMaterial = worldMaterial

        if solids:
            for solid in solids:
                self.add(solid)

        self.resetOutsideMaterial()

    def add(self, solid: Solid, position: Vector = None):
        if position:
            solid.translateTo(position)
        self._validateLabel(solid)
        if not self._ignoreIntersections:
            self._validatePosition(solid)
        self._solids.append(solid)

    @property
    def solids(self):
        return self._solids

    def addToViewer(self, viewer: Abstract3DViewer, representation="surface", colormap="bone", opacity=0.1, **kwargs):
        viewer.add(*self.solids, representation=representation, colormap=colormap, opacity=opacity, **kwargs)

    def getWorldEnvironment(self) -> Environment:
        return Environment(self._worldMaterial)

    def _validatePosition(self, newSolid: Solid):
        """Assert newSolid position is valid and make proper adjustments so that the
        material at each solid interface is well-defined."""
        if len(self._solids) == 0:
            return

        intersectingSuspects = self._findIntersectingSuspectsFor(newSolid)
        if len(intersectingSuspects) == 0:
            return

        intersectingSuspects.sort(key=lambda s: s.getBoundingBox().xMax - s.getBoundingBox().xMin, reverse=True)

        for otherSolid in intersectingSuspects:
            if newSolid.contains(*otherSolid.getVertices()):
                self._processContainedSolid(otherSolid, container=newSolid)
                break
            elif otherSolid.contains(*newSolid.getVertices()):
                self._processContainedSolid(newSolid, container=otherSolid)
            else:
                raise NotImplementedError(
                    "Cannot place a solid that partially intersects with an existing solid. "
                    "Since this might be underestimating containment, you can also create a "
                    "scene with 'ignoreIntersections=True' to ignore this error and manually "
                    "handle environments of contained solids with "
                    "containedSolid.setOutsideEnvironment(containerSolid.getEnvironment())."
                )

    def _processContainedSolid(self, solid: Solid, container: Solid):
        if container.isStack():
            containerEnv = self._getEnvironmentOfStackAt(solid.position, container)
        else:
            containerEnv = container.getEnvironment()
        solid.setOutsideEnvironment(containerEnv)

        containerLabel = containerEnv.solid.getLabel()
        if containerLabel not in self._solidsContainedIn:
            self._solidsContainedIn[containerLabel] = [solid.getLabel()]
        else:
            self._solidsContainedIn[containerLabel].append(solid.getLabel())

    def _validateLabel(self, solid):
        labelSet = set(s.getLabel() for s in self.solids)
        if solid.getLabel() not in labelSet:
            return
        idx = 2
        while f"{solid.getLabel()}_{idx}" in labelSet:
            idx += 1
        warnings.warn(
            f"A solid with label '{solid.getLabel()}' already exists in the scene. "
            f"Renaming to '{solid.getLabel()}_{idx}'."
        )
        solid.setLabel(f"{solid.getLabel()}_{idx}")

    def _findIntersectingSuspectsFor(self, solid) -> List[Solid]:
        solidBBox = solid.getBoundingBox()
        intersectingSuspects = []
        for otherSolid in self._solids:
            if solidBBox.intersects(otherSolid.getBoundingBox()):
                intersectingSuspects.append(otherSolid)
        return intersectingSuspects

    def getSolids(self) -> List[Solid]:
        return self._solids

    def getSolid(self, solidLabel: str) -> Solid:
        for solid in self._solids:
            if solid.getLabel().lower() == solidLabel.lower():
                return solid
            if not solid.isStack():
                continue
            for layerLabel in solid.getLayerLabels():
                if layerLabel.lower() == solidLabel.lower():
                    return solid
        raise ValueError(f"Solid '{solidLabel}' not found in scene. Available solids: {self.getSolidLabels()}")

    def getSolidLabels(self) -> List[str]:
        labels = []
        for solid in self._solids:
            if solid.isStack():
                labels.extend(solid.getLayerLabels())
            else:
                labels.append(solid.getLabel())
        return labels

    def getSurfaceLabels(self, solidLabel) -> List[str]:
        solid = self.getSolid(solidLabel)
        if solid is None:
            return []
        if solid.isStack() and solid.getLabel() != solidLabel:
            return solid.getLayerSurfaceLabels(solidLabel)
        return solid.surfaceLabels

    def getContainedSolidLabels(self, solidLabel: str) -> List[str]:
        return self._solidsContainedIn.get(solidLabel, [])

    def getPolygons(self) -> List[Polygon]:
        polygons = []
        for solid in self._solids:
            polygons.extend(solid.getPolygons())
        return polygons

    def getMaterials(self) -> list:
        materials = [self._worldMaterial]
        for solid in self._solids:
            surfaceLabels = solid.surfaceLabels
            for surfaceLabel in surfaceLabels:
                material = solid.getPolygons(surfaceLabel)[0].insideEnvironment.material
                if material not in materials:
                    materials.append(material)
        return list(materials)

    def getMaterial(self, solidLabel: str):
        solid = self.getSolid(solidLabel)
        if solid.isStack():
            layerSurfaceLabel = [s for s in solid.getLayerSurfaceLabels(solidLabel) if INTERFACE_KEY not in s][0]
            return solid.getEnvironment(layerSurfaceLabel).material
        else:
            return solid.getEnvironment().material

    def getBoundingBox(self) -> Optional[BoundingBox]:
        if len(self._solids) == 0:
            return None

        bbox = self._solids[0].getBoundingBox().copy()
        for solid in self._solids[1:]:
            bbox.extendTo(solid.getBoundingBox())
        return bbox

    def resetOutsideMaterial(self):
        outsideEnvironment = self.getWorldEnvironment()
        for solid in self._solids:
            if self._isHidden(solid.getLabel()):
                continue
            solid.setOutsideEnvironment(outsideEnvironment)

    def _isHidden(self, solidLabel: str) -> bool:
        for hiddenLabels in self._solidsContainedIn.values():
            if solidLabel in hiddenLabels:
                return True
        return False

    def getEnvironmentAt(self, position: Vector) -> Environment:
        # First, recursively look if position is in a contained solid.
        for containerLabel in self._solidsContainedIn.keys():
            env = self._getEnvironmentOfContainerAt(position, containerLabel)
            if env is not None:
                return env

        for solid in self._solids:
            if solid.contains(position):
                if solid.isStack():
                    return self._getEnvironmentOfStackAt(position, solid)
                return solid.getEnvironment()
        return self.getWorldEnvironment()

    def _getEnvironmentOfContainerAt(self, position: Vector, containerLabel: str) -> Optional[Environment]:
        containerSolid = self.getSolid(containerLabel)
        if not containerSolid.contains(position):
            return None
        for containedLabel in self.getContainedSolidLabels(containerLabel):
            containedEnv = self._getEnvironmentOfContainerAt(position, containedLabel)
            if containedEnv:
                return containedEnv
        if containerSolid.isStack():
            return self._getEnvironmentOfStackAt(position, containerSolid)
        return containerSolid.getEnvironment()

    @staticmethod
    def _getEnvironmentOfStackAt(position: Vector, stack: Solid) -> Environment:
        """Returns the environment of the stack at the given position.

        To do that we first find the interface in the stack that is closest to the given position.
        At the same time we find on which side of the interface we are and return the environment
        of this side from any surface polygon.
        """
        environment = None
        closestDistance = sys.maxsize
        for surfaceLabel in stack.surfaceLabels:
            if INTERFACE_KEY not in surfaceLabel:
                continue
            planePolygon = stack.surfaces.getPolygons(surfaceLabel)[0]
            planeNormal = planePolygon.normal
            planePoint = planePolygon.vertices[0]
            v = position - planePoint
            distanceFromPlane = v.dot(planeNormal)
            if abs(distanceFromPlane) < closestDistance:
                closestDistance = abs(distanceFromPlane)
                isInside = distanceFromPlane < 0
                environment = planePolygon.insideEnvironment if isInside else planePolygon.outsideEnvironment
        return environment

    def __hash__(self):
        solidHash = hash(tuple(sorted([hash(s) for s in self._solids])))
        worldMaterialHash = hash(self._worldMaterial) if self._worldMaterial else 0
        return hash((solidHash, worldMaterialHash))
