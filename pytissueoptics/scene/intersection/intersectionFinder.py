import math
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

from pytissueoptics.scene import Material
from pytissueoptics.scene.geometry import Vector, Polygon
from pytissueoptics.scene.intersection import Ray
from pytissueoptics.scene.tree import SpacePartition, Node
from pytissueoptics.scene.tree.treeConstructor.binary import NoSplitThreeAxesConstructor
from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.intersection.bboxIntersect import GemsBoxIntersect
from pytissueoptics.scene.intersection.mollerTrumboreIntersect import MollerTrumboreIntersect
from pytissueoptics.scene.solids import Solid


@dataclass
class IntersectionInfo:
    distance: float
    position: Vector = None
    polygon: Polygon = None


@dataclass
class Intersection:
    distance: float
    position: Vector
    normal: Vector
    insideMaterial: Material
    outsideMaterial: Material
    distanceLeft: float = None


class IntersectionFinder:
    def __init__(self, scene: Scene):
        self._scene = scene
        self._polygonIntersect = MollerTrumboreIntersect()
        self._boxIntersect = GemsBoxIntersect()

    def findIntersection(self, ray: Ray) -> Optional[Intersection]:
        raise NotImplementedError

    def _findClosestPolygonIntersection(self, ray: Ray, polygons: List[Polygon]) -> Optional[IntersectionInfo]:
        closestIntersectionInfo = IntersectionInfo(sys.maxsize)
        for polygon in polygons:
            intersectionPoint = self._polygonIntersect.getIntersection(ray, polygon)
            if not intersectionPoint:
                continue
            distance = (intersectionPoint - ray.origin).getNorm()
            if distance < closestIntersectionInfo.distance:
                closestIntersectionInfo = IntersectionInfo(distance, intersectionPoint, polygon)

        if closestIntersectionInfo.distance == sys.maxsize:
            return None
        return closestIntersectionInfo

    def _makeIntersection(self, ray: Ray, intersectionInfo: IntersectionInfo) -> Optional[Intersection]:
        if not intersectionInfo:
            return None
        polygon = intersectionInfo.polygon
        intersection = Intersection(intersectionInfo.distance, intersectionInfo.position,
                                    polygon.normal, polygon.insideMaterial, polygon.outsideMaterial)
        if ray.length is not None:
            intersection.distanceLeft = ray.length - intersectionInfo.distance
        if polygon.toSmooth:
            intersection.normal = self._getSmoothNormal(polygon, intersection.position)
        return intersection

    def _getSmoothNormal(self, polygon, position):
        """ If the intersecting polygon was prepared for smoothing (ie. it has vertex
        normals), we interpolate the normal at the intersection point using the normal
        of all its vertices. The interpolation is done using the general barycentric
        coordinates algorithm from http://www.geometry.caltech.edu/pubs/MHBD02.pdfv. """
        # todo: move to shader module and test
        weights = []
        n = len(polygon.vertices)
        for i, vertex in enumerate(polygon.vertices):
            prevVertex = polygon.vertices[(i - 1) % n]
            nextVertex = polygon.vertices[(i + 1) % n]
            w = (self._cotangent(position, vertex, prevVertex) +
                 self._cotangent(position, vertex, nextVertex)) / (position - vertex).getNorm() ** 2
            weights.append(w)
        weights = [w/sum(weights) for w in weights]

        smoothNormal = Vector(0, 0, 0)
        for weight, vertex in zip(weights, polygon.vertices):
            smoothNormal += weight * vertex.normal
        smoothNormal.normalize()

        return smoothNormal

    @staticmethod
    def _cotangent(a, b, c):
        """ Cotangent of triangle abc at vertex b. """
        ba = a - b
        bc = c - b
        return bc.dot(ba) / bc.cross(ba).getNorm()


class SimpleIntersectionFinder(IntersectionFinder):
    def findIntersection(self, ray: Ray) -> Optional[Intersection]:
        bboxIntersections = self._findBBoxIntersectingSolids(ray)
        bboxIntersections.sort(key=lambda x: x[0])
        for (distance, solid) in bboxIntersections:
            intersectionInfo = self._findClosestPolygonIntersection(ray, solid.getPolygons())
            if intersectionInfo:
                return self._makeIntersection(ray, intersectionInfo)
        return None

    def _findBBoxIntersectingSolids(self, ray) -> Optional[List[Tuple[float, Solid]]]:
        """ We need to handle the special case where ray starts inside bbox. The Box Intersect will not compute
        the intersection for this case and will instead return ray.origin. When that happens, distance will be 0
        and we exit to check the polygons of this solid. """
        solidCandidates = []
        for solid in self._scene.solids:
            bboxIntersectionPoint = self._boxIntersect.getIntersection(ray, solid.bbox)
            if not bboxIntersectionPoint:
                continue
            distance = (bboxIntersectionPoint - ray.origin).getNorm()
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
        intersectionInfo = self._findIntersection(ray, self._partition.root)
        return self._makeIntersection(ray, intersectionInfo)

    def _findIntersection(self, ray: Ray, node: Node, closestDistance=sys.maxsize) -> Optional[IntersectionInfo]:
        if node.isLeaf:
            intersectionInfo = self._findClosestPolygonIntersection(ray, node.polygons)
            return intersectionInfo

        if not self._nodeIsWorthExploring(ray, node, closestDistance):
            return None

        closestIntersectionInfo = None
        for child in node.children:
            intersectionInfo = self._findIntersection(ray, child, closestDistance)
            if intersectionInfo is None:
                continue
            if intersectionInfo.distance < closestDistance:
                closestDistance = intersectionInfo.distance
                closestIntersectionInfo = intersectionInfo

        return closestIntersectionInfo

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
