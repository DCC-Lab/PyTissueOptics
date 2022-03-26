import math
from typing import List
import numpy as np
from pytissueoptics.scene import Vector
from pytissueoptics.scene.intersection import Ray


class RaySource:
    def __init__(self):
        self._rays: List[Ray] = []
        self._createRays()

    @property
    def rays(self):
        return self._rays

    def _createRays(self):
        raise NotImplementedError


class UniformRaySource(RaySource):
    def __init__(self, position: Vector, direction: Vector, xTheta: float, yTheta: float, xResolution: int, yResolution: int):
        self._position = position
        self._direction = direction
        self._direction.normalize()
        self._theta = (xTheta * np.pi / 180, yTheta * np.pi / 180)
        self._resolution = (xResolution, yResolution)
        super(UniformRaySource, self).__init__()

    def _createRays(self):
        for yTheta in self._getYThetaRange():
            for xTheta in self._getXThetaRange():
                self._createRayAt(xTheta, yTheta)

    def _getXThetaRange(self) -> List[float]:
        return np.linspace(0, self._theta[0], self._resolution[0]) - self._theta[0] / 2

    def _getYThetaRange(self) -> List[float]:
        return np.linspace(0, self._theta[1], self._resolution[1]) - self._theta[1] / 2

    def _createRayAt(self, xTheta: float, yTheta: float):
        self._rays.append(Ray(self._position, self._getRayDirectionAt(xTheta, yTheta)))

    def _getRayDirectionAt(self, xTheta, yTheta) -> Vector:
        """
        Returns the (normalized) direction of the ray at the given x and y angle difference
        from the source orientation.

        xTheta is defined as the angle from -Z axis towards +X axis.
        yTheta is defined as the angle from -Z axis towards +Y axis.
        """
        xTheta += math.atan(self._direction.x / self._direction.z)
        yTheta += math.asin(self._direction.y)
        rayDirection = Vector(-math.sin(xTheta)*math.cos(yTheta), math.sin(yTheta), -math.cos(xTheta)*math.cos(yTheta))
        return rayDirection
