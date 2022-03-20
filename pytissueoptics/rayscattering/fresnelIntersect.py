import math
import random
from typing import Optional

from pytissueoptics.scene import Vector, Material
from pytissueoptics.scene.intersection.intersectionFinder import Intersection


class FresnelIntersect:
    def __init__(self, rayDirection: Vector, intersection: Intersection):
        surface = intersection.polygon
        rayDirection = rayDirection
        normal = surface.normal.copy()

        goingInside = rayDirection.dot(surface.normal) < 0
        if goingInside:
            normal.multiply(-1)
            self._indexIn = surface.outsideMaterial.index
            self._indexOut = surface.insideMaterial.index
            self._nextMaterial = surface.insideMaterial
        else:
            self._indexIn = surface.insideMaterial.index
            self._indexOut = surface.outsideMaterial.index
            self._nextMaterial = surface.outsideMaterial

        self._incidencePlane = rayDirection.cross(normal)
        if self._incidencePlane.getNorm() < 1e-7:
            self._incidencePlane = rayDirection.anyPerpendicular()
        else:
            self._incidencePlane.normalize()

        self._thetaIn = math.acos(normal.dot(rayDirection))
        self._thetaOut = None

    @property
    def nextMaterial(self) -> Optional[Material]:
        return self._nextMaterial

    @property
    def incidencePlane(self) -> Vector:
        return self._incidencePlane

    def reflectionCoefficient(self, theta) -> float:
        """ Fresnel reflection coefficient, directly from MCML code in
        Wang, L-H, S.L. Jacques, L-Q Zheng:
        MCML - Monte Carlo modeling of photon transport in multi-layered
        tissues. Computer Methods and Programs in Biomedicine 47:131-146, 1995.

        """
        n1 = self._indexIn
        n2 = self._indexOut

        if n1 == n2:
            return 0

        if theta == 0:
            R = (n2-n1)/(n2+n1)
            return R*R

        sa1 = math.sin(theta)
        if sa1*n1/n2 > 1:
            return 1

        sa2 = sa1*n1/n2
        ca1 = math.sqrt(1-sa1*sa1)
        if 1-sa2*sa2 > 0:
            ca2 = math.sqrt(1-sa2*sa2)
        else:
            ca2 = 0
        cap = ca1*ca2 - sa1*sa2  # c+ = cc - ss.
        cam = ca1*ca2 + sa1*sa2  # c- = cc + ss.
        sap = sa1*ca2 + ca1*sa2  # s+ = sc + cs.
        sam = sa1*ca2 - ca1*sa2  # s- = sc - cs.
        r = 0.5*sam*sam*(cam*cam+cap*cap)/(sap*sap*cam*cam)
        return r

    def isReflected(self) -> bool:
        R = self.reflectionCoefficient(self._thetaIn)
        if random.random() < R:
            return True
        return False

    @property
    def reflectionDeflection(self) -> float:
        return 2 * self._thetaIn - math.pi

    @property
    def refractionDeflection(self) -> float:
        sinThetaOut = self._indexIn * math.sin(self._thetaIn) / self._indexOut

        # todo: remove this debug case when tested.
        if abs(sinThetaOut) > 1:
            # We should not be here.
            raise ValueError("Can't refract beyond angle of total reflection")

        self._thetaOut = math.asin(sinThetaOut)
        return self._thetaIn - self._thetaOut
