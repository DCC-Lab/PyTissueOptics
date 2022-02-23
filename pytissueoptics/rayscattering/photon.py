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
            self._weight -= 0.1  # fixme: safety condition
            self._roulette()

    def _step(self, distance):
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
        # todo
        # self.ez.rotateAround(intersection.polygon.normal, np.random.rand())
        # self.ez.rotateAround(intersection.polygon.normal, self._getRefractionAngle(intersection))

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

    # def _getRefractionAngle(self, intersection):
    #     surfaceNormal = intersection.polygon.normal
    #     if intersection.polygon.normal.cross(self._direction) > 0:
    #         thetaIn = math.acos(surfaceNormal.dot(self._direction))
    #         inMaterial = intersection.polygon.outsideMaterial
    #         outMaterial = intersection.polygon.insideMaterial
    #     else:
    #         thetaIn = math.acos(-surfaceNormal.dot(self._direction))
    #         inMaterial = intersection.polygon.insideMaterial
    #         outMaterial = intersection.polygon.outsideMaterial
    #
    #     sinThetaOut = inMaterial.index*math.sin(thetaIn)/outMaterial.index
    #     if abs(sinThetaOut) > 1:
    #         # We should not be here.
    #         raise ValueError("Can't refract beyond angle of total reflection")
    #
    #     thetaOut = np.arcsin(sinThetaOut)
    #     return thetaIn - thetaOut
