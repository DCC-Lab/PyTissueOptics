import math

import numpy as np

from pytissueoptics.scene import Vector, Material
from pytissueoptics.scene.intersection import Ray
from pytissueoptics.scene.intersection.intersectionFinder import IntersectionFinder, Intersection
from pytissueoptics.scene.logger import Logger


class Photon:
    def __init__(self, position: Vector, direction: Vector):
        self._position = position
        self._direction = direction
        self._weight = 1

        self._intersectionFinder = None
        self._material = None
        self._logger = None
        self._worldMaterial = None

    @property
    def _isAlive(self) -> bool:
        return self._weight > 0

    def setContext(self, worldMaterial: Material, intersectionFinder: IntersectionFinder, logger: Logger):
        self._worldMaterial = worldMaterial
        # todo: set proper initial material
        self._material = worldMaterial
        self._intersectionFinder = intersectionFinder
        self._logger = logger

    def propagate(self):
        self._logger.logPoint(self._position)
        while self._isAlive:
            distance = self._material.getScatteringDistance()
            self._step(distance)
            self._roulette()

    def _step(self, distance):
        # TODO: reflect, scatter, finish distance
        stepRay = Ray(self._position, self._direction, distance)
        intersection = self._intersectionFinder.findIntersection(stepRay)

        if intersection:
            self._logger.logPoint(intersection.position)
            self._position += self._direction * (intersection.distance + 1e-3)

            self._refract(intersection)
            self._updateMaterial(self._getNextMaterial(intersection))

        elif self._material.isVacuum:
            self._weight = 0

        else:
            self._position += self._direction * distance
            delta = self._weight * self._material.albedo
            self._decreaseWeightBy(delta)

    def _refract(self, intersection):
        surfaceNormal = intersection.polygon.normal

        goingInside = surfaceNormal.dot(self._direction) < 0
        if goingInside:
            surfaceNormal *= -1

        incidencePlane = self._direction.cross(surfaceNormal)
        self._rotateAround(incidencePlane, self._getRefractionAngle(intersection, goingInside))

        if self._logger:
            self._logger.logPoint(self._position)

    def _getNextMaterial(self, intersection: Intersection):
        if intersection.polygon.normal.dot(self._direction) > 0:
            return intersection.polygon.outsideMaterial
        else:
            return intersection.polygon.insideMaterial

    def _updateMaterial(self, material):
        if material is None:
            self._material = self._worldMaterial
        else:
            self._material = material

    def _decreaseWeightBy(self, delta):
        if self._logger:
            self._logger.logDataPoint(delta, position=self._position)
        self._weight -= delta
        if self._weight < 0:
            self.weight = 0

    def _roulette(self):
        chance = 0.1
        if self._weight >= 1e-4 or self._weight == 0:
            return
        elif np.random.random() < chance:
            # todo: are we gonna talk about this?
            self._weight /= chance
        else:
            self._weight = 0

    def _getRefractionAngle(self, intersection: Intersection, goingInside: bool):
        surfaceNormal = intersection.polygon.normal

        if goingInside:
            thetaIn = math.acos(-surfaceNormal.dot(self._direction))
            inMaterial = intersection.polygon.outsideMaterial
            outMaterial = intersection.polygon.insideMaterial
        else:
            thetaIn = math.acos(surfaceNormal.dot(self._direction))
            inMaterial = intersection.polygon.insideMaterial
            outMaterial = intersection.polygon.outsideMaterial

        sinThetaOut = inMaterial.index * math.sin(thetaIn) / outMaterial.index
        if abs(sinThetaOut) > 1:
            # We should not be here.
            raise ValueError("Can't refract beyond angle of total reflection")

        thetaOut = np.arcsin(sinThetaOut)
        return thetaIn - thetaOut

    def _rotateAround(self, unitAxis: Vector, theta: float):
        # This is the most expensive (and most common)
        # operation when performing Monte Carlo in tissue
        # (15% of time spent here). It is difficult to optimize without
        # making it even less readable than it currently is
        # http://en.wikipedia.org/wiki/Rotation_matrix
        #
        # Several options were tried in the past such as
        # external not-so-portable C library, unreadable
        # shortcuts, sine and cosine lookup tables, etc...
        # and the performance gain was minimal (<20%).
        # For now, this is the best, most readable solution.

        cost = math.cos(theta)
        sint = math.sin(theta)
        one_cost = 1 - cost

        ux = unitAxis.x
        uy = unitAxis.y
        uz = unitAxis.z

        X = self._direction.x
        Y = self._direction.y
        Z = self._direction.z

        x = (cost + ux * ux * one_cost) * X \
            + (ux * uy * one_cost - uz * sint) * Y \
            + (ux * uz * one_cost + uy * sint) * Z
        y = (uy * ux * one_cost + uz * sint) * X \
            + (cost + uy * uy * one_cost) * Y \
            + (uy * uz * one_cost - ux * sint) * Z
        z = (uz * ux * one_cost - uy * sint) * X \
            + (uz * uy * one_cost + ux * sint) * Y \
            + (cost + uz * uz * one_cost) * Z

        self._direction.update(x, y, z)
