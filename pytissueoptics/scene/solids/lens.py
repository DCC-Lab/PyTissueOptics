import math
from typing import List
import itertools

import numpy as np

from pytissueoptics.scene.geometry import Vector, primitives, Vertex
from pytissueoptics.scene.solids import Cylinder
from pytissueoptics.scene.material import RefractiveMaterial


class ThickLens(Cylinder):
    """
    The Lens is defined by a front radius, a back radius, a diameter and a thickness (along its center).
    A positive frontRadius means that the front surface is convex. This is reversed for the back surface.
    A flat surface is obtained by setting the corresponding radius to 0 or math.inf.
    The position refers to the vector from global origin to its centroid.
    The generated mesh will be divided into the following subgroups: Front, Side and Back.
    By default, front will point towards the negative z-axis.
    """
    def __init__(self, frontRadius: float, backRadius: float, diameter: float, thickness: float,
                 position: Vector = Vector(0, 0, 0), material=None, label: str = "thick lens",
                 primitive: str = primitives.DEFAULT, smooth: bool = True, u: int = 32, v: int = 2, s: int = 10):
        frontRadius = frontRadius if frontRadius != 0 else math.inf
        backRadius = backRadius if backRadius != 0 else math.inf
        if abs(frontRadius) <= diameter / 2:
            raise ValueError(f"Front radius must be greater than the lens radius. Front radius: {frontRadius}, "
                             f"lens radius: {diameter / 2}")
        if abs(backRadius) <= diameter / 2:
            raise ValueError(f"Back radius must be greater than the lens radius. Back radius: {backRadius}, "
                             f"lens radius: {diameter / 2}")

        self._diameter = diameter
        self._frontRadius = frontRadius
        self._backRadius = backRadius
        self._centerThickness = thickness

        length = self._computeLateralLength()
        super().__init__(radius=diameter / 2, length=length, u=u, v=v, s=s, position=position,
                         material=material, label=label, primitive=primitive, smooth=smooth)

    @property
    def _hasFrontCurvature(self) -> bool:
        return self._frontRadius != 0 and self._frontRadius != math.inf

    @property
    def _hasBackCurvature(self) -> bool:
        return self._backRadius != 0 and self._backRadius != math.inf

    def _computeLateralLength(self) -> float:
        """ Returns the thickness of the lens on its side. This is the length required to build the base cylinder
        before applying the curvature. """
        dt1, dt2 = 0, 0
        if self._hasFrontCurvature:
            dt1 = abs(self._frontRadius) - math.sqrt(self._frontRadius**2 - self._diameter**2/4)
            dt1 *= np.sign(self._frontRadius)
        if self._hasBackCurvature:
            dt2 = abs(self._backRadius) - math.sqrt(self._backRadius**2 - self._diameter**2/4)
            dt2 *= -np.sign(self._backRadius)
        return max(self._centerThickness - dt1 - dt2, 0)

    @property
    def focalLength(self) -> float:
        """ Returns the focal length of the lens in air. Requires a refractive material to be defined."""
        if self._material is None or not issubclass(type(self._material), RefractiveMaterial):
            raise ValueError("Cannot compute focal length without refractive material defined.")
        # For thick lenses, the focal length is given by the lensmaker's equation:
        # 1/f = (n - 1) * (1/R1 - 1/R2 + (n - 1) * d / (n * R1 * R2))
        n = self._material.n
        R1 = self._frontRadius
        R2 = self._backRadius
        d = self._centerThickness
        if n == 1:
            return math.inf
        if self._hasFrontCurvature and self._hasBackCurvature:
            return 1 / ((n - 1) * (1 / R1 - 1 / R2 + (n - 1) * d / (n * R1 * R2)))
        if self._hasFrontCurvature:
            return R1 / (n - 1)
        if self._hasBackCurvature:
            return R2 / (n - 1)
        return math.inf

    def _computeVerticesOfLayers(self) -> tuple:
        frontLayers, lateralLayers, backLayers = super()._computeVerticesOfLayers()

        if self._hasFrontCurvature:
            frontVertices = list(itertools.chain.from_iterable(frontLayers)) + [self._frontCenter]
            self._applyCurvature(self._frontRadius, frontVertices)
        if self._hasBackCurvature:
            backVertices = list(itertools.chain.from_iterable(backLayers)) + [self._backCenter]
            self._applyCurvature(self._backRadius, backVertices)

        if self._frontCenter.z > self._backCenter.z:
            raise ValueError("Not a valid lens: curved surfaces intersect.")
        return frontLayers, lateralLayers, backLayers

    def _applyCurvature(self, radius: float, vertices: List[Vertex]):
        # At this point, all vertices are on the same z plane.
        surfaceZ = vertices[0].z
        # The sphere origin is simply found by setting the z coordinate so that the distance to
        # the surface perimeter equals the desired radius.
        sphereOrigin = Vector(0, 0, surfaceZ + math.sqrt(radius**2 - self._radius**2) * np.sign(radius))
        for vertex in vertices:
            direction = vertex - sphereOrigin
            direction.normalize()
            vertex.update(*(sphereOrigin + direction * abs(radius)).array)

    def smooth(self, surfaceLabel: str = None, reset: bool = True):
        if surfaceLabel:
            return super(Cylinder, self).smooth(surfaceLabel, reset)
        for surfaceLabel in self._surfaces.surfaceLabels:
            self.smooth(surfaceLabel, reset=False)

