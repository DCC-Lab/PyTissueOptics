import math
import random
from typing import Optional

from pytissueoptics.rayscattering.fresnel import FresnelIntersect
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
        self._hasContext = False

    @property
    def isAlive(self) -> bool:
        return self._weight > 0

    @property
    def position(self) -> Vector:
        return self._position

    @property
    def direction(self) -> Vector:
        return self._direction

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def material(self) -> Material:
        return self._material

    def setContext(self, worldMaterial: Material, intersectionFinder: IntersectionFinder = None, logger: Logger = None):
        # todo: set proper initial material
        self._worldMaterial = worldMaterial
        self._material = worldMaterial
        self._intersectionFinder = intersectionFinder
        # todo: PhotonLogger with logEventX booleans:
        #  logInitialPositions, logIntersections, logScattering, logEndPositions
        self._logger = logger
        self._hasContext = True

    def propagate(self):
        if not self._hasContext:
            raise NotImplementedError("Cannot propagate photon without context. Use ‘setContext(...)‘. ")

        self._logPosition()
        while self.isAlive:
            distance = self._material.getScatteringDistance()
            self.step(distance)
            self._roulette()

    def step(self, distance):
        intersection = self._getIntersection(distance)

        if intersection:
            self.moveBy(intersection.distance)
            self._logPosition()
            distanceLeft = distance - intersection.distance

            fresnelIntersection = FresnelIntersect(self._direction, intersection)
            if fresnelIntersection.isReflected():
                self.reflect(fresnelIntersection)
            else:
                self.refract(fresnelIntersection)
                self._updateMaterial(fresnelIntersection.nextMaterial)
                newDistance = self._material.getScatteringDistance()
                distanceLeft *= newDistance / distance

            self.moveBy(1e-3)
            self.step(distanceLeft)

        elif self._material.isVacuum:
            self._weight = 0

        else:
            self.moveBy(distance)
            self.scatter()

    def _getIntersection(self, distance) -> Optional[Intersection]:
        if self._intersectionFinder is None:
            return None
        stepRay = Ray(self._position, self._direction, distance)
        return self._intersectionFinder.findIntersection(stepRay)

    def moveBy(self, distance):
        self._position += self._direction * distance

    def reflect(self, fresnelIntersection: FresnelIntersect):
        # todo: replace with simple dot product logic ?
        #  Reflection = Incidence - Normal * 2 * Incidence.dot(Normal)
        self._direction.rotateAround(fresnelIntersection.incidencePlane,
                                     fresnelIntersection.reflectionDeflection)

    def refract(self, fresnelIntersection: FresnelIntersect):
        self._direction.rotateAround(fresnelIntersection.incidencePlane,
                                     fresnelIntersection.refractionDeflection)

    def _updateMaterial(self, material):
        if material is None:
            self._material = self._worldMaterial
        else:
            self._material = material

    def scatter(self):
        delta = self._weight * self._material.albedo
        self._decreaseWeightBy(delta)
        theta, phi = self._material.getScatteringAngles()
        self._scatterBy(theta, phi)

    def _decreaseWeightBy(self, delta):
        self._logWeightDecrease(delta)
        self._weight -= delta
        if self._weight < 0:
            self._weight = 0

    def _scatterBy(self, theta, phi):
        self._er.rotateAround(self._direction, phi)
        self._direction.rotateAround(self._er, theta)

    def _roulette(self):
        chance = 0.1
        if self._weight >= 1e-4 or self._weight == 0:
            return
        elif random.random() < chance:
            # todo: are we gonna talk about this?
            self._weight /= chance
        else:
            self._weight = 0

    def _logPosition(self):
        if self._logger is not None:
            self._logger.logPoint(self._position)

    def _logWeightDecrease(self, delta):
        if self._logger:
            self._logger.logDataPoint(delta, position=self._position)
