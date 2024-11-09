from typing import Union, Optional

from pytissueoptics.scene.geometry import Vector, Triangle, Quad, Polygon
from pytissueoptics.scene.intersection import Ray


EPS_CORRECTION = 0.0005


class MollerTrumboreIntersect:
    EPS_CATCH = 0.000001
    EPS_PARALLEL = 0.00001
    EPS_SIDE = 0.000001

    def getIntersection(self, ray: Ray, polygon: Union[Triangle, Quad, Polygon]) -> Optional[Vector]:
        if isinstance(polygon, Triangle):
            return self._getTriangleIntersection(ray, polygon)
        if isinstance(polygon, Quad):
            return self._getQuadIntersection(ray, polygon)
        if isinstance(polygon, Polygon):
            return self._getPolygonIntersection(ray, polygon)

    def _getTriangleIntersection(self, ray: Ray, triangle: Triangle) -> Optional[Vector]:
        """ Möller–Trumbore ray-triangle 3D intersection algorithm.
        Added epsilon zones to avoid numerical errors in the OpenCL implementation.
        Modified to support rays with finite length:
            A. If the intersection is too far away, return None.
            B. If the intersection is just a bit too far away, there is no intersection, but we cannot accept
                the ray to land there (it would be too close to the surface and might yield intersection errors
                later on). Therefore, we tag the intersection as 'tooClose' and use that later to move the ray's
                landing position a bit away from this surface.
        """
        v1, v2, v3 = triangle.vertices
        edgeA = v2 - v1
        edgeB = v3 - v1
        pVector = ray.direction.cross(edgeB)
        determinant = edgeA.dot(pVector)

        rayIsParallel = abs(determinant) < self.EPS_PARALLEL
        if rayIsParallel:
            return None

        inverseDeterminant = 1. / determinant
        tVector = ray.origin - v1
        u = tVector.dot(pVector) * inverseDeterminant
        if u < -self.EPS_SIDE or u > 1.:
            # EPS_SIDE is used to make the triangle a bit larger than it is
            # to be sure a ray could not sneak between two triangles.
            return None

        qVector = tVector.cross(edgeA)
        v = ray.direction.dot(qVector) * inverseDeterminant
        if v < -self.EPS_SIDE or u + v > 1.:
            return None

        # Distance to intersection point
        t = edgeB.dot(qVector) * inverseDeterminant
        if t > 0 and (ray.length > t or ray.length is None):
            # Case 1: Trivial case. Intersects.
            return ray.origin + ray.direction * t

        # Next we need to check if the intersection is inside the epsilon catch zone (forward or backward).
        # Note that this mechanic only works when same-solid intersections are ignored before calling this function.
        dt = t - ray.length
        dt_T = abs(triangle.normal.dot(ray.direction) * dt)
        if t > ray.length and dt_T < self.EPS_CATCH:
            # Case 2: Forward epsilon catch. Ray ends close to the triangle, so we intersect at the ray's end.
            return ray.origin + ray.direction * ray.length
        if t < 0 and dt_T < self.EPS_CATCH:
            # Case 3: Backward epsilon catch. Ray starts close to the triangle, so we intersect at the origin.
            # This requires the intersector to always test triangles (or at least, close ones) of the origin solid.
            return ray.origin

        # Case 4: No intersection.
        return None

    def _getQuadIntersection(self, ray: Ray, quad: Quad) -> Optional[Vector]:
        v1, v2, v3, v4 = quad.vertices
        triangleA = Triangle(v1, v2, v4)
        triangleB = Triangle(v2, v3, v4)
        intersectionA = self._getTriangleIntersection(ray, triangleA)
        if intersectionA:
            return intersectionA
        return self._getTriangleIntersection(ray, triangleB)

    def _getPolygonIntersection(self, ray: Ray, polygon: Polygon) -> Optional[Vector]:
        for i in range(len(polygon.vertices) - 2):
            triangle = Triangle(polygon.vertices[0], polygon.vertices[i + 1], polygon.vertices[i + 2])
            intersection = self.getIntersection(ray, triangle)
            if intersection:
                return intersection
        return None
