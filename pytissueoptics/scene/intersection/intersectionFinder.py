import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

from pytissueoptics.scene.geometry import Vector, Polygon
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
    distanceLeft: float = None


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

        if closestIntersection.distance == sys.maxsize:
            return None
        if ray.length is not None:
            closestIntersection.distanceLeft = ray.length - closestIntersection.distance
        return closestIntersection


class SimpleIntersectionFinder(IntersectionFinder):
    def findIntersection(self, ray: Ray) -> Optional[Intersection]:
        bboxIntersections = self._findBBoxIntersectingSolids(ray)
        bboxIntersections.sort(key=lambda x: x[0])
        for (distance, solid) in bboxIntersections:
            intersection = self._findClosestPolygonIntersection(ray, solid.getPolygons())
            if intersection:
                return intersection
        return None

    def _findBBoxIntersectingSolids(self, ray) -> Optional[List[Tuple[float, Solid]]]:
        """ We need to handle the special case where ray starts inside bbox. The Box Intersect will not compute
        the intersection for this case and will instead return ray.origin. When that happens, distance will be 0
        and we exit to check the polygons of this solid. """
        solidCandidates = []
        for solid in self._scene.solids:
            bboxIntersection = self._boxIntersect.getIntersection(ray, solid.bbox)
            if not bboxIntersection:
                continue
            distance = (bboxIntersection - ray.origin).getNorm()
            solidCandidates.append((distance, solid))
            if distance == 0:
                break
        return solidCandidates


class FastIntersectionFinder(IntersectionFinder):
    def __init__(self, scene: Scene, constructor=NoSplitThreeAxesConstructor(), maxDepth=20, minLeafSize=6):
        super(FastIntersectionFinder, self).__init__(scene)
        self._partition = SpacePartition(self._scene.getBoundingBox(), self._scene.getPolygons(), constructor,
                                         maxDepth, minLeafSize)

    def findIntersection(self, ray: Ray) -> Optional[Intersection]:
        intersection = self._findIntersection(ray, self._partition.root)
        self._smooth(intersection)
        return intersection

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
        bboxIntersection = self._boxIntersect.getIntersection(ray, node.bbox)
        if bboxIntersection is None:
            return False
        bboxDistance = (bboxIntersection - ray.origin).getNorm()
        if bboxDistance > closestDistance:
            return False
        return True

    @property
    def partition(self):
        return self._partition
