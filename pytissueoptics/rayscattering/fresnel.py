import math
import random
from dataclasses import dataclass

from pytissueoptics.scene.geometry import Environment, Vector
from pytissueoptics.scene.intersection.intersectionFinder import Intersection


@dataclass
class FresnelIntersection:
    nextEnvironment: Environment
    incidencePlane: Vector
    isReflected: bool
    angleDeflection: float


class FresnelIntersect:
    _indexIn: float
    _indexOut: float
    _thetaIn: float

    def compute(self, rayDirection: Vector, intersection: Intersection) -> FresnelIntersection:
        rayDirection = rayDirection
        normal = intersection.normal.copy()

        goingInside = rayDirection.dot(normal) < 0
        if goingInside:
            normal.multiply(-1)
            self._indexIn = intersection.outsideEnvironment.material.n
            self._indexOut = intersection.insideEnvironment.material.n
            nextEnvironment = intersection.insideEnvironment
        else:
            self._indexIn = intersection.insideEnvironment.material.n
            self._indexOut = intersection.outsideEnvironment.material.n
            nextEnvironment = intersection.outsideEnvironment

        incidencePlane = rayDirection.cross(normal)
        if incidencePlane.getNorm() < 1e-7:
            incidencePlane = rayDirection.getAnyOrthogonal()
        incidencePlane.normalize()

        dot = normal.dot(rayDirection)
        dot = max(min(dot, 1), -1)
        self._thetaIn = math.acos(dot)

        return self._create(nextEnvironment, incidencePlane)

    def _create(self, nextEnvironment, incidencePlane) -> FresnelIntersection:
        reflected = self._getIsReflected()
        if reflected:
            angleDeflection = self._getReflectionDeflection()
        else:
            angleDeflection = self._getRefractionDeflection()

        return FresnelIntersection(nextEnvironment, incidencePlane, reflected, angleDeflection)

    def _getIsReflected(self) -> bool:
        R = self._getReflectionCoefficient()
        if random.random() < R:
            return True
        return False

    def _getReflectionCoefficient(self) -> float:
        """ Fresnel reflection coefficient, directly from MCML code in
        Wang, L-H, S.L. Jacques, L-Q Zheng:
        MCML - Monte Carlo modeling of photon transport in multi-layered
        tissues. Computer Methods and Programs in Biomedicine 47:131-146, 1995.
        """

        n1 = self._indexIn
        n2 = self._indexOut

        if n1 == n2:
            return 0

        if self._thetaIn == 0:
            R = (n2-n1)/(n2+n1)
            return R*R

        sa1 = math.sin(self._thetaIn)

        sa2 = sa1 * n1 / n2
        if sa2 >= 1:
            return 1

        ca1 = math.sqrt(1-sa1*sa1)
        ca2 = math.sqrt(1-sa2*sa2)

        cap = ca1*ca2 - sa1*sa2  # c+ = cc - ss.
        cam = ca1*ca2 + sa1*sa2  # c- = cc + ss.
        sap = sa1*ca2 + ca1*sa2  # s+ = sc + cs.
        sam = sa1*ca2 - ca1*sa2  # s- = sc - cs.
        r = 0.5*sam*sam*(cam*cam+cap*cap)/(sap*sap*cam*cam)
        return r

    def _getReflectionDeflection(self) -> float:
        return 2 * self._thetaIn - math.pi

    def _getRefractionDeflection(self) -> float:
        sinThetaOut = self._indexIn * math.sin(self._thetaIn) / self._indexOut
        thetaOut = math.asin(sinThetaOut)
        return self._thetaIn - thetaOut
