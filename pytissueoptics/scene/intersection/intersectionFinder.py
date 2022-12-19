import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

from pytissueoptics.scene import shader
from pytissueoptics.scene.geometry import Vector, Polygon, Environment
from pytissueoptics.scene.intersection import Ray
from pytissueoptics.scene.tree import SpacePartition, Node
from pytissueoptics.scene.tree.treeConstructor.binary import NoSplitThreeAxesConstructor
from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.intersection.bboxIntersect import GemsBoxIntersect
from pytissueoptics.scene.intersection.mollerTrumboreIntersect import MollerTrumboreIntersect
from pytissueoptics.scene.solids import Solid


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
    isTooClose: bool = False


class IntersectionFinder:
    def __init__(self, scene: Scene):
        self._scene = scene
        self._polygonIntersect = MollerTrumboreIntersect()
        self._boxIntersect = GemsBoxIntersect()

    def findIntersection(self, ray: Ray) -> Optional[Intersection]:
        raise NotImplementedError

    def _findClosestPolygonIntersection(self, ray: Ray, polygons: List[Polygon]) -> Optional[Intersection]:
        closestIntersection = Intersection(sys.maxsize)
        for polygon in polygons:
            intersectionPoint = self._polygonIntersect.getIntersection(ray, polygon)
            if not intersectionPoint:
                continue
            distance = (intersectionPoint - ray.origin).getNorm()
            if distance < closestIntersection.distance:
                closestIntersection = Intersection(distance, intersectionPoint, polygon)
                closestIntersection.isTooClose = ray.isTooClose

        if closestIntersection.distance == sys.maxsize:
            return None
        return closestIntersection

    @staticmethod
    def _composeIntersection(ray: Ray, intersection: Intersection) -> Optional[Intersection]:
        if not intersection:
            return None

        intersection.normal = shader.getSmoothNormal(intersection.polygon, intersection.position)
        intersection.insideEnvironment = intersection.polygon.insideEnvironment
        intersection.outsideEnvironment = intersection.polygon.outsideEnvironment
        intersection.surfaceLabel = intersection.polygon.surfaceLabel

        if ray.length is not None:
            intersection.distanceLeft = ray.length - intersection.distance

        return intersection


class SimpleIntersectionFinder(IntersectionFinder):
    def findIntersection(self, ray: Ray) -> Optional[Intersection]:
        """
        Find the closest intersection between a ray and the scene.

        1. Find all solids for which the ray intersects their bounding box.
            N.B.: Note that the ray may start inside the bounding box. In which case the bbox intersection test is not
                possible, and we simply set the bbox intersection distance to zero.
        2. Sort these solid candidates by bbox intersection distance.
        3. For each solid, find the closest polygon intersection.
            3.1 If a polygon intersection is found for a given solid candidate, then we return this intersection point
            without testing the other solid candidates since they are ordered by distance.
            N.B.: Except for the case of multiple contained solids where it is not possible to order them in a
            meaningful way, so in this case we need to test all of them (candidate distance is zero) before
            returning the closest intersection found.
        """
        bboxIntersections = self._findBBoxIntersectingSolids(ray)
        bboxIntersections.sort(key=lambda x: x[0])

        closestDistance = sys.maxsize
        closestIntersection = None
        for i, (distance, solid) in enumerate(bboxIntersections):
            contained = distance == 0
            if not contained and closestIntersection:
                break
            intersection = self._findClosestPolygonIntersection(ray, solid.getPolygons())
            if intersection and intersection.distance < closestDistance:
                closestDistance = intersection.distance
                closestIntersection = intersection

        return self._composeIntersection(ray, closestIntersection)

    def _findBBoxIntersectingSolids(self, ray) -> Optional[List[Tuple[float, Solid]]]:
        """ We need to handle the special case where ray starts inside bbox. The Box Intersect will not compute
        the intersection for this case and will instead return ray.origin. When that happens, distance will be 0,
        and we continue to check for possibly other contained solids. """
        solidCandidates = []
        for solid in self._scene.solids:
            bboxIntersectionPoint = self._boxIntersect.getIntersection(ray, solid.bbox)
            if not bboxIntersectionPoint:
                continue
            distance = (bboxIntersectionPoint - ray.origin).getNorm()
            solidCandidates.append((distance, solid))
        return solidCandidates


class FastIntersectionFinder(IntersectionFinder):
    def __init__(self, scene: Scene, constructor=NoSplitThreeAxesConstructor(), maxDepth=20, minLeafSize=6):
        super(FastIntersectionFinder, self).__init__(scene)
        self._partition = SpacePartition(self._scene.getBoundingBox(), self._scene.getPolygons(), constructor,
                                         maxDepth, minLeafSize)

    def findIntersection(self, ray: Ray) -> Optional[Intersection]:
        intersection = self._findIntersection(ray, self._partition.root)
        return self._composeIntersection(ray, intersection)

    def _findIntersection(self, ray: Ray, node: Node, closestDistance=sys.maxsize) -> Optional[Intersection]:
        if node.isLeaf:
            intersection = self._findClosestPolygonIntersection(ray, node.polygons)
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

    @property
    def partition(self):
        return self._partition
