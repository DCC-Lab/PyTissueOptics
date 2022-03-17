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

        self._er = self._direction.anyPerpendicular()
        self._er.normalize()

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
            self._scatter()

    def _refract(self, intersection):
        surfaceNormal = intersection.polygon.normal

        goingInside = surfaceNormal.dot(self._direction) < 0
        if goingInside:
            surfaceNormal *= -1

        incidencePlane = self._direction.cross(surfaceNormal)
        self._direction.rotateAround(incidencePlane, self._getRefractionAngle(intersection, goingInside))

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

    def _scatter(self):
        delta = self._weight * self._material.albedo
        self._decreaseWeightBy(delta)
        theta, phi = self._material.getScatteringAngles()
        self._scatterBy(theta, phi)

    def _decreaseWeightBy(self, delta):
        if self._logger:
            self._logger.logDataPoint(delta, position=self._position)
        self._weight -= delta
        if self._weight < 0:
            self.weight = 0

    def _scatterBy(self, theta, phi):
        self._er.rotateAround(self._direction, phi)
        self._direction.rotateAround(self._er, theta)

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
