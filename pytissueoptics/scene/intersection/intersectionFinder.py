import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

from pytissueoptics.scene import shader
from pytissueoptics.scene.geometry import Environment, Polygon, Vector
from pytissueoptics.scene.geometry.polygon import WORLD_LABEL
from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.tree import Node, SpacePartition
from pytissueoptics.scene.tree.treeConstructor.binary import NoSplitThreeAxesConstructor

from .bboxIntersect import GemsBoxIntersect
from .mollerTrumboreIntersect import MollerTrumboreIntersect
from .ray import Ray


@dataclass
class Intersection:
    distance: float
    position: Vector = None
    polygon: Polygon = None
    normal: Vector = None
    insideEnvironment: Environment = None
    outsideEnvironment: Environment = None
    surfaceLabel: str = None
    distanceLeft: float = None
    isSmooth: bool = False
    rawNormal: Vector = None


class IntersectionFinder:
    def __init__(self, scene: Scene):
        self._scene = scene
        self._polygonIntersect = MollerTrumboreIntersect()
        self._boxIntersect = GemsBoxIntersect()

    def findIntersection(self, ray: Ray, currentSolidLabel: Optional[str]) -> Optional[Intersection]:
        raise NotImplementedError

    def _findClosestPolygonIntersection(
        self, ray: Ray, polygons: List[Polygon], currentSolidLabel: str
    ) -> Optional[Intersection]:
        closestIntersection = Intersection(sys.maxsize)
        minSameSolidDistance = -sys.maxsize

        for polygon in polygons:
            # When an interface joins a side surface, an outside photon could try to intersect with the interface.
            #  This is not allowed, so we skip these tests (where surface environments dont match the photon).
            insideLabel = polygon.insideEnvironment.solidLabel
            outsideLabel = polygon.outsideEnvironment.solidLabel if polygon.outsideEnvironment else WORLD_LABEL
            if insideLabel != currentSolidLabel and outsideLabel != currentSolidLabel:
                continue

            intersectionPoint = self._polygonIntersect.getIntersection(ray, polygon)
            if not intersectionPoint:
                continue
            distance = (intersectionPoint - ray.origin).getNorm()

            # Discard intersection result if the ray is heading towards its current solid
            # (possible because of the epsilon catch zone in our Moller-Trumbore intersect).
            isGoingInside = ray.direction.dot(polygon.normal) < 0
            nextSolidLabel = insideLabel if isGoingInside else outsideLabel
            if nextSolidLabel == currentSolidLabel:
                minSameSolidDistance = max(minSameSolidDistance, distance)
                continue

            if abs(distance) < abs(closestIntersection.distance):
                closestIntersection = Intersection(distance, intersectionPoint, polygon)

        if closestIntersection.distance == sys.maxsize:
            return None

        if closestIntersection.distance == 0 and minSameSolidDistance == 0:
            # Cancel back catch. Surface overlap.
            return None
        if closestIntersection.distance < 0 and minSameSolidDistance > closestIntersection.distance + 1e-7:
            # Cancel back catch if the same-solid intersect distance is greater.
            return None

        return closestIntersection

    @staticmethod
    def _composeIntersection(ray: Ray, intersection: Intersection) -> Optional[Intersection]:
        if not intersection:
            return None

        intersection.insideEnvironment = intersection.polygon.insideEnvironment
        intersection.outsideEnvironment = intersection.polygon.outsideEnvironment
        intersection.surfaceLabel = intersection.polygon.surfaceLabel
        intersection.rawNormal = intersection.polygon.normal

        if ray.length is not None:
            intersection.distanceLeft = ray.length - intersection.distance

        smoothNormal = shader.getSmoothNormal(intersection.polygon, intersection.position)

        # If the resulting smooth normal changes the sign of the dot product with the ray direction, do not smooth.
        if smoothNormal.dot(ray.direction) * intersection.polygon.normal.dot(ray.direction) < 0:
            intersection.normal = intersection.polygon.normal
            intersection.isSmooth = False
            return intersection

        intersection.normal = smoothNormal
        intersection.isSmooth = True
        return intersection


class SimpleIntersectionFinder(IntersectionFinder):
    def findIntersection(self, ray: Ray, currentSolidLabel: str) -> Optional[Intersection]:
        """
        Find the closest intersection between a ray and the scene.

        1. Find all solids for which the ray intersects their bounding box.
            N.B.: Note that the ray may start inside the bounding box. In which case the bbox intersection test is not
                possible, and we simply set the bbox intersection distance to zero.
        2. Sort these solid candidates by bbox intersection distance.
        3. For each solid, find the closest polygon intersection.
            If the solid bbox distance is greater than the closest intersection distance, then we can stop testing since
            they are ordered by distance. Note that bbox distance is zero if the ray starts inside.
        """
        bboxIntersections = self._findBBoxIntersectingSolids(ray, currentSolidLabel)
        bboxIntersections.sort(key=lambda x: x[0])

        closestDistance = sys.maxsize
        closestIntersection = None
        for i, (distance, solid) in enumerate(bboxIntersections):
            if distance > closestDistance:
                break
            intersection = self._findClosestPolygonIntersection(ray, solid.getPolygons(), currentSolidLabel)
            if intersection and intersection.distance < closestDistance:
                closestDistance = intersection.distance
                closestIntersection = intersection

        return self._composeIntersection(ray, closestIntersection)

    def _findBBoxIntersectingSolids(self, ray: Ray, currentSolidLabel: str) -> Optional[List[Tuple[float, Solid]]]:
        """We need to handle the special case where ray starts inside bbox. The Box Intersect will not compute
        the intersection for this case and will instead return ray.origin. When that happens, distance will be 0,
        and we continue to check for possibly other contained solids."""
        solidCandidates = []
        for solid in self._scene.solids:
            if solid.getLabel() == currentSolidLabel:
                bboxIntersectionPoint = ray.origin
            else:
                bboxIntersectionPoint = self._boxIntersect.getIntersection(ray, solid.bbox)
            if not bboxIntersectionPoint:
                continue
            distance = (bboxIntersectionPoint - ray.origin).getNorm()
            solidCandidates.append((distance, solid))
        return solidCandidates


class FastIntersectionFinder(IntersectionFinder):
    def __init__(self, scene: Scene, constructor=NoSplitThreeAxesConstructor(), maxDepth=20, minLeafSize=6):
        super(FastIntersectionFinder, self).__init__(scene)
        self._partition = SpacePartition(
            self._scene.getBoundingBox(), self._scene.getPolygons(), constructor, maxDepth, minLeafSize
        )

    def findIntersection(self, ray: Ray, currentSolidLabel: str) -> Optional[Intersection]:
        self._currentSolidLabel = currentSolidLabel
        intersection = self._findIntersection(ray, self._partition.root)
        return self._composeIntersection(ray, intersection)

    def _findIntersection(self, ray: Ray, node: Node, closestDistance=sys.maxsize) -> Optional[Intersection]:
        # todo: test if this node search is still compatible with the new core intersection logic
        #  which can now require testing a polygon slightly behind the ray origin.
        if node.isLeaf:
            intersection = self._findClosestPolygonIntersection(ray, node.polygons, self._currentSolidLabel)
            return intersection

        if not self._nodeIsWorthExploring(ray, node, closestDistance):
            return None

        closestIntersection = None
        for child in node.children:
            intersection = self._findIntersection(ray, child, closestDistance)
            if intersection is None:
                continue
            if intersection.distance < closestDistance:
                closestDistance = intersection.distance
                closestIntersection = intersection

        return closestIntersection

    def _nodeIsWorthExploring(self, ray, node, closestDistance) -> bool:
        bboxIntersectionPoint = self._boxIntersect.getIntersection(ray, node.bbox)
        if bboxIntersectionPoint is None:
            return False
        bboxDistance = (bboxIntersectionPoint - ray.origin).getNorm()
        if bboxDistance > closestDistance:
            return False
        return True
