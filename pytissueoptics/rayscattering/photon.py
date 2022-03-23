import math
import random
from typing import Optional

from pytissueoptics.rayscattering.fresnel import FresnelIntersectionFactory, FresnelIntersection
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
        self._worldMaterial = None
        self._logger = None
        self._fresnelIntersectionFactory = None

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

    def setContext(self, worldMaterial: Material, intersectionFinder: IntersectionFinder = None, logger: Logger = None,
                   fresnelIntersectionFactory=FresnelIntersectionFactory()):
        # todo: set proper initial material
        self._worldMaterial = worldMaterial
        self._material = worldMaterial
        self._intersectionFinder = intersectionFinder
        # todo: PhotonLogger with logEventX booleans:
        #  logInitialPositions, logIntersections, logScattering, logEndPositions
        self._logger = logger
        self._hasContext = True
        self._fresnelIntersectionFactory = fresnelIntersectionFactory

    def propagate(self):
        if not self._hasContext:
            raise NotImplementedError("Cannot propagate photon without context. Use ‘setContext(...)‘. ")

        self._logPosition()
        distance = 0
        while self.isAlive:
            distance = self.step(distance)
            self.roulette()

    def step(self, distance) -> float:
        if distance == 0:
            distance = self._material.getScatteringDistance()

        intersection = self._getIntersection(distance)

        if intersection:
            self.moveBy(intersection.distance)
            self._logPosition()
            distanceLeft = self.reflectOrRefract(intersection)
        else:
            if math.isinf(distance):
                self._weight = 0
                return 0

            self.moveBy(distance)
            distanceLeft = 0

            self.scatter()

        return distanceLeft

    def _getIntersection(self, distance) -> Optional[Intersection]:
        if self._intersectionFinder is None:
            return None

        stepRay = Ray(self._position, self._direction, distance)
        return self._intersectionFinder.findIntersection(stepRay)

    def reflectOrRefract(self, intersection: Intersection):
        fresnelIntersection = self._getFresnelIntersection(intersection)

        if fresnelIntersection.isReflected:
            self.reflect(fresnelIntersection)
        else:
            self.refract(fresnelIntersection)

            mus1 = self._material.mu_s
            mus2 = fresnelIntersection.nextMaterial.mu_s
            if mus1 == 0:
                intersection.distanceLeft = 0
            elif mus2 != 0:
                intersection.distanceLeft *= mus1/mus2
            else:
                intersection.distanceLeft = math.inf

            self._updateMaterial(fresnelIntersection.nextMaterial)

        return intersection.distanceLeft

    def _getFresnelIntersection(self, intersection: Intersection) -> FresnelIntersection:
        return self._fresnelIntersectionFactory.compute(self._direction, intersection)

    def moveBy(self, distance):
        self._position += self._direction * distance

    def reflect(self, fresnelIntersection: FresnelIntersection):
        # todo: replace with simple dot product logic ?
        #  Reflection = Incidence - Normal * 2 * Incidence.dot(Normal)
        self._direction.rotateAround(fresnelIntersection.incidencePlane,
                                     fresnelIntersection.angleDeflection)

    def refract(self, fresnelIntersection: FresnelIntersection):
        self._direction.rotateAround(fresnelIntersection.incidencePlane,
                                     fresnelIntersection.angleDeflection)

    def _updateMaterial(self, material):
        if material is None:
            self._material = self._worldMaterial
        else:
            self._material = material

    def scatter(self):
        theta, phi = self._material.getScatteringAngles()
        self.scatterBy(theta, phi)
        self.interact()

    def scatterBy(self, theta, phi):
        self._er.rotateAround(self._direction, phi)
        self._direction.rotateAround(self._er, theta)

    def interact(self):
        delta = self._weight * self._material.albedo
        self._decreaseWeightBy(delta)

    def _decreaseWeightBy(self, delta):
        self._logWeightDecrease(delta)
        self._weight -= delta
        if self._weight < 0:
            self._weight = 0

    def roulette(self):
        chance = 0.1
        if self._weight >= 1e-4 or self._weight == 0:
            return
        elif random.random() < chance:
            self._weight /= chance
        else:
            self._weight = 0

    def _logPosition(self):
        if self._logger is not None:
            self._logger.logPoint(self._position)

    def _logWeightDecrease(self, delta):
        if self._logger:
            self._logger.logDataPoint(delta, position=self._position)
