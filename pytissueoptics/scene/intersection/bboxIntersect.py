from typing import Union

from pytissueoptics.scene.geometry import BoundingBox, Vector
from pytissueoptics.scene.intersection import Ray


class BoundingBoxIntersect:
    def intersect(self, ray: Ray, bbox: BoundingBox) -> Union[Vector, None]:
        raise NotImplemented


class GemsBoxIntersect(BoundingBoxIntersect):
    """ Graphics Gems Fast Ray-Box Intersection """
    pass


class ZacharBoxIntersect(BoundingBoxIntersect):
    """ https://gamedev.stackexchange.com/a/18459 """
    def intersect(self, ray: Ray, bbox: BoundingBox) -> Union[Vector, None]:
        inverseDirection = self._safeInverse(ray.direction)
        minCorner = Vector(bbox.xMin, bbox.yMin, bbox.zMin)
        maxCorner = Vector(bbox.xMax, bbox.yMax, bbox.zMax)

        t1 = (minCorner.x - ray.origin.x) * inverseDirection.x
        t2 = (maxCorner.x - ray.origin.x) * inverseDirection.x

        t3 = (minCorner.y - ray.origin.y) * inverseDirection.y
        t4 = (maxCorner.y - ray.origin.y) * inverseDirection.y

        t5 = (minCorner.z - ray.origin.z) * inverseDirection.z
        t6 = (maxCorner.z - ray.origin.z) * inverseDirection.z

        tMin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6))
        tMax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6))
        if tMax < 0:
            return None

        if tMin > tMax:
            return None

        t = tMin
        return ray.origin + t * ray.direction

    @staticmethod
    def _safeInverse(direction: Vector) -> Vector:
        epsilon = 1.0 * 10 ** (-37)
        x, y, z = direction.array
        if x == 0.0:
            x += epsilon
        if y == 0.0:
            y += epsilon
        if z == 0.0:
            z += epsilon
        return Vector(1.0/x, 1.0/y, 1.0/z)
