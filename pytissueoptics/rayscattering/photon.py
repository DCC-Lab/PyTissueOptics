import math
import random
from typing import Optional

from pytissueoptics.rayscattering.fresnel import FresnelIntersect, FresnelIntersection
from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.scene.geometry import Environment, Vector
from pytissueoptics.scene.intersection import Ray
from pytissueoptics.scene.intersection.intersectionFinder import IntersectionFinder, Intersection
from pytissueoptics.scene.intersection.mollerTrumboreIntersect import EPS_CORRECTION
from pytissueoptics.scene.logger import Logger, InteractionKey

WORLD_LABEL = "world"
WEIGHT_THRESHOLD = 1e-4


class Photon:
    def __init__(self, position: Vector, direction: Vector):
        self._position = position
        self._direction = direction
        self._weight = 1
        self._environment: Environment = None

        self._er = self._direction.getAnyOrthogonal()
        self._er.normalize()
        self._hasContext = False
        self._fresnelIntersect: FresnelIntersect = None

        self._intersectionFinder: Optional[IntersectionFinder] = None
        self._logger: Optional[Logger] = None

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
    def material(self) -> ScatteringMaterial:
        return self._environment.material

    @property
    def solidLabel(self):
        if not self._environment.solid:
            return WORLD_LABEL
        return self._environment.solid.getLabel()

    def setContext(self, environment: Environment, intersectionFinder: IntersectionFinder = None, logger: Logger = None,
                   fresnelIntersect=FresnelIntersect()):
        self._environment: Environment = environment
        self._intersectionFinder = intersectionFinder
        self._logger = logger
        self._hasContext = True
        self._fresnelIntersect = fresnelIntersect

    def propagate(self):
        if not self._hasContext:
            raise NotImplementedError("Cannot propagate photon without context. Use ‘setContext(...)‘. ")

        distance = 0
        while self.isAlive:
            distance = self.step(distance)
            self.roulette()

    def step(self, distance=0) -> float:
        if distance == 0:
            distance = self.material.getScatteringDistance()

        intersection = self._getIntersection(distance)

        if intersection and not intersection.isTooClose:
            self.moveBy(intersection.distance)
            distanceLeft = self.reflectOrRefract(intersection)
        else:
            if math.isinf(distance):
                self._weight = 0
                return 0

            self.moveBy(distance)
            distanceLeft = 0

            if intersection and intersection.isTooClose:
                # Photon will land too close to the surface, so we need to move it away from the surface.
                stepSign = 1
                solidTowardsNormal = intersection.outsideEnvironment.solid
                if solidTowardsNormal != self._environment.solid:
                    stepSign = -1
                stepCorrection = intersection.normal * stepSign * EPS_CORRECTION
                self._position += stepCorrection

            self.scatter()

        return distanceLeft

    def _getIntersection(self, distance) -> Optional[Intersection]:
        if self._intersectionFinder is None:
            return None

        stepRay = Ray(self._position, self._direction, distance)
        return self._intersectionFinder.findIntersection(stepRay)

    def reflectOrRefract(self, intersection: Intersection):
        fresnelIntersection = self._getFresnelIntersection(intersection)

        # Determine required step sign to move away from intersecting surface
        stepSign = 1
        solidTowardsNormal = intersection.outsideEnvironment.solid
        if solidTowardsNormal != self._environment.solid:
            stepSign = -1
        if not fresnelIntersection.isReflected:
            stepSign *= -1

        if fresnelIntersection.isReflected:
            self.reflect(fresnelIntersection)
        else:
            self._logIntersection(intersection)
            self.refract(fresnelIntersection)

            mut1 = self.material.mu_t
            mut2 = fresnelIntersection.nextEnvironment.material.mu_t
            if mut1 == 0:
                intersection.distanceLeft = 0
            elif mut2 != 0:
                intersection.distanceLeft *= mut1/mut2
            else:
                intersection.distanceLeft = math.inf

            self._environment = fresnelIntersection.nextEnvironment

        # Move away from intersecting surface by a small amount
        stepCorrection = intersection.normal * stepSign * EPS_CORRECTION
        self._position += stepCorrection

        # Remove this distance correction from the distance left, but set to zero if the result is negative.
        intersection.distanceLeft -= EPS_CORRECTION
        if intersection.distanceLeft < 0:
            intersection.distanceLeft = 0

        return intersection.distanceLeft

    def _getFresnelIntersection(self, intersection: Intersection) -> FresnelIntersection:
        return self._fresnelIntersect.compute(self._direction, intersection)

    def moveBy(self, distance):
        self._position += self._direction * distance

    def reflect(self, fresnelIntersection: FresnelIntersection):
        self._direction.rotateAround(fresnelIntersection.incidencePlane,
                                     fresnelIntersection.angleDeflection)

    def refract(self, fresnelIntersection: FresnelIntersection):
        self._direction.rotateAround(fresnelIntersection.incidencePlane,
                                     fresnelIntersection.angleDeflection)

    def scatter(self):
        theta, phi = self.material.getScatteringAngles()
        self.scatterBy(theta, phi)
        self.interact()

    def scatterBy(self, theta, phi):
        self._er.rotateAround(self._direction, phi)
        self._direction.rotateAround(self._er, theta)

    def interact(self):
        delta = self._weight * self.material.getAlbedo()
        self._decreaseWeightBy(delta)

    def _decreaseWeightBy(self, delta):
        self._logWeightDecrease(delta)
        self._weight -= delta

    def roulette(self):
        chance = 0.1
        if self._weight >= WEIGHT_THRESHOLD or self._weight == 0:
            return
        elif random.random() < chance:
            self._weight /= chance
        else:
            self._weight = 0

    def _logIntersection(self, intersection: Intersection):
        if self._logger is None:
            return
        solidA = intersection.insideEnvironment.solid
        solidLabelA = solidA.getLabel() if solidA else None
        key = InteractionKey(solidLabelA, intersection.surfaceLabel)
        isLeavingSurface = self._direction.dot(intersection.normal) > 0
        sign = 1 if isLeavingSurface else -1
        self._logger.logDataPoint(sign * self._weight, self._position, key)

        solidB = intersection.outsideEnvironment.solid
        if solidB is None:
            return
        solidLabelB = solidB.getLabel()
        key = InteractionKey(solidLabelB, intersection.surfaceLabel)
        self._logger.logDataPoint(-sign * self._weight, self._position, key)

    def _logWeightDecrease(self, delta):
        if self._logger:
            key = InteractionKey(self.solidLabel)
            self._logger.logDataPoint(delta, self._position, key)
