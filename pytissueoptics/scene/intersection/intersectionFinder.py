import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

from pytissueoptics.scene.geometry import Vector, Polygon, Triangle, Quad
from pytissueoptics.scene.intersection import Ray
from pytissueoptics.scene.tree import SpacePartition, Node
from pytissueoptics.scene.tree.treeConstructor.binary import SAHWideAxisTreeConstructor
from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.intersection.bboxIntersect import GemsBoxIntersect
from pytissueoptics.scene.intersection.quadIntersect import MollerTrumboreQuadIntersect
from pytissueoptics.scene.intersection.triangleIntersect import MollerTrumboreTriangleIntersect
from pytissueoptics.scene.solids import Solid


@dataclass
class Intersection:
    distance: float
    position: Vector
    polygon: Polygon


class IntersectionFinder:
    def __init__(self, scene: Scene):
        self._scene = scene
        self._triangleIntersect = MollerTrumboreTriangleIntersect()
        self._quadIntersect = MollerTrumboreQuadIntersect()
        self._boxIntersect = GemsBoxIntersect()

    def findIntersection(self, ray: Ray) -> Optional[Intersection]:
        raise NotImplementedError

    def _findClosestPolygonIntersection(self, ray: Ray, polygons: List[Polygon]) -> Optional[Intersection]:
        closestPolygon = None
        closestIntersection = None
        closestDistance = sys.maxsize
        for polygon in polygons:
            intersection = self._findPolygonIntersection(ray, polygon)
            if not intersection:
                continue
            distance = (intersection - ray.origin).getNorm()
            if distance < closestDistance:
                closestDistance = distance
                closestIntersection = intersection
                closestPolygon = polygon
        if not closestIntersection:
            return None
        return Intersection(closestDistance, closestIntersection, closestPolygon)

    def _findPolygonIntersection(self, ray: Ray, polygon: Polygon) -> Optional[Vector]:
        if isinstance(polygon, Triangle):
            return self._triangleIntersect.getIntersection(ray, polygon)
        if isinstance(polygon, Quad):
            return self._quadIntersect.getIntersection(ray, polygon)


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
    def __init__(self, scene: Scene, constructor=SAHWideAxisTreeConstructor(), maxDepth=20, minLeafSize=6):
        super(FastIntersectionFinder, self).__init__(scene)
        self._partition = SpacePartition(self._scene.getBoundingBox(), self._scene.getPolygons(), constructor,
                                         maxDepth, minLeafSize)

    def findIntersection(self, ray: Ray) -> Optional[Intersection]:
        """
        This is a simple home-made algorithm.

        It is part of a recursive backtrack algorithm family. First, the ray origin is found with
        a recursive point search. Then, the intersection is found by checking is the ray intersects the neighbour
        node bbox. If it does, this node is explored. When a leaf node is reached, the polygons within that leaf node
        are explored for a ray-polygon intersection. If a hit is computed, it continues to backtrack until a closer
        bbox it hit. If the algorithm backtrack all the way up to the root without a hit, the algorithm stops.

        Limitations:    - does not take in consideration if the touched polygon is shared amongst many nodes
        """

        # rayStartingNode = self._partition.searchPoint(ray.origin)
        # if rayStartingNode is None:
        #     if not self._boxIntersect.getIntersection(ray, self._partition.root.bbox):
        #         return None
        rayStartingNode = self._partition.root
        intersection = self._exploreNodeForIntersection(ray, rayStartingNode)
        return intersection

    def _exploreNodeForIntersection(self, ray: Ray, node: Node, closestIntersection=None) -> Optional[Intersection]:
        """Will only try to intersect polygons when it reached a leaf node."""
        closestIntersection = closestIntersection
        if node.isLeaf:
            intersection = self._findClosestPolygonIntersection(ray, node.polygons)
            # node.visited = True
            # if intersection is None:
            #     intersection = self._exploreNodeForIntersection(ray, node.parent)
            return intersection

        for child in node.children:
            bboxIntersection = self._boxIntersect.getIntersection(ray, child.bbox)
            if bboxIntersection is None:
                continue
            if closestIntersection is not None:
                if (bboxIntersection - ray.origin).getNorm() > closestIntersection.distance:
                    continue
            intersection = self._exploreNodeForIntersection(ray, child)
            if intersection is None:
                continue
            if closestIntersection is None:
                closestIntersection = intersection
            elif intersection.distance < closestIntersection.distance:
                closestIntersection = intersection

        # if not node.isRoot:
        #     self._exploreNodeForIntersection(ray, node.parent)

        return closestIntersection

