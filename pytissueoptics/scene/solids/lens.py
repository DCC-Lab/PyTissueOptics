import itertools
import math
from typing import List

import numpy as np

from pytissueoptics.scene.geometry import Vector, Vertex, primitives
from pytissueoptics.scene.material import RefractiveMaterial
from pytissueoptics.scene.solids import Cylinder


class ThickLens(Cylinder):
    """
    The Lens is defined by a front radius, a back radius, a diameter and a thickness (along its center).
    A positive frontRadius means that the front surface is convex. This is reversed for the back surface.
    A flat surface is obtained by setting the corresponding radius to 0 or math.inf.
    The position refers to the vector from global origin to its centroid.
    The generated mesh will be divided into the following subgroups: Front, Side and Back.
    By default, front will point towards the negative z-axis.
    """

    def __init__(
        self,
        frontRadius: float,
        backRadius: float,
        diameter: float,
        thickness: float,
        position: Vector = Vector(0, 0, 0),
        material=None,
        label: str = "thick lens",
        primitive: str = primitives.DEFAULT,
        smooth: bool = True,
        u: int = 24,
        v: int = 2,
        s: int = 24,
    ):
        frontRadius = frontRadius if frontRadius != 0 else math.inf
        backRadius = backRadius if backRadius != 0 else math.inf
        if abs(frontRadius) <= diameter / 2:
            raise ValueError(
                f"Front radius must be greater than the lens radius. Front radius: {frontRadius}, "
                f"lens radius: {diameter / 2}"
            )
        if abs(backRadius) <= diameter / 2:
            raise ValueError(
                f"Back radius must be greater than the lens radius. Back radius: {backRadius}, "
                f"lens radius: {diameter / 2}"
            )

        self._diameter = diameter
        self._frontRadius = frontRadius
        self._backRadius = backRadius
        self._thickness = thickness

        length = self._computeEdgeThickness(thickness)
        super().__init__(
            radius=diameter / 2,
            length=length,
            u=u,
            v=v,
            s=s,
            position=position,
            material=material,
            label=label,
            primitive=primitive,
            smooth=smooth,
        )

    @property
    def _hasFrontCurvature(self) -> bool:
        return self._frontRadius != 0 and self._frontRadius != math.inf

    @property
    def _hasBackCurvature(self) -> bool:
        return self._backRadius != 0 and self._backRadius != math.inf

    def _computeEdgeThickness(self, centerThickness) -> float:
        """Returns the thickness of the lens on its side. This is the length required to build the base cylinder
        before applying surface curvature."""
        dt1, dt2 = 0, 0
        if self._hasFrontCurvature:
            dt1 = abs(self._frontRadius) - math.sqrt(self._frontRadius**2 - self._diameter**2 / 4)
            dt1 *= np.sign(self._frontRadius)
        if self._hasBackCurvature:
            dt2 = abs(self._backRadius) - math.sqrt(self._backRadius**2 - self._diameter**2 / 4)
            dt2 *= -np.sign(self._backRadius)
        edgeThickness = centerThickness - dt1 - dt2
        if edgeThickness < 0:
            raise ValueError("Desired center thickness is too small for the given radii and diameter.")
        return edgeThickness

    @property
    def centerThickness(self) -> float:
        return (self._frontCenter - self._backCenter).getNorm()

    @property
    def edgeThickness(self) -> float:
        return self._length

    @property
    def frontRadius(self) -> float:
        return self._frontRadius

    @property
    def backRadius(self) -> float:
        return self._backRadius

    @property
    def focalLength(self) -> float:
        """Returns the focal length of the lens in air. Requires a refractive material to be defined."""
        if self._material is None or not issubclass(type(self._material), RefractiveMaterial):
            raise ValueError("Cannot compute focal length without refractive material defined.")
        # For thick lenses, the focal length is given by the lensmaker's equation:
        # 1/f = (n - 1) * (1/R1 - 1/R2 + (n - 1) * d / (n * R1 * R2))
        n = self._material.n
        R1 = self._frontRadius
        R2 = self._backRadius
        d = self.centerThickness
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
        for surfaceLabel in ["front", "back"]:
            self.smooth(surfaceLabel, reset=False)

    def _geometryParams(self) -> dict:
        return {
            "diameter": self._diameter,
            "frontRadius": self._frontRadius,
            "backRadius": self._backRadius,
            "thickness": self._thickness,
            "length": self._length,
            "u": self._u,
            "v": self._v,
            "s": self._s,
        }


class SymmetricLens(ThickLens):
    """A symmetrical thick lens of focal length `f` in air."""

    def __init__(
        self,
        f: float,
        diameter: float,
        thickness: float,
        material: RefractiveMaterial,
        position: Vector = Vector(0, 0, 0),
        label: str = "lens",
        primitive: str = primitives.DEFAULT,
        smooth: bool = True,
        u: int = 24,
        v: int = 2,
        s: int = 24,
    ):
        # For thick lenses, the focal length is given by the lensmaker's equation:
        # 1/f = (n - 1) * (1/R1 - 1/R2 + (n - 1) * d / (n * R1 * R2))
        # with R2 = -R1, we get the following quadratic equation to solve:
        # 1/f = (n - 1) * (2/R - (n - 1) * d / (n * R^2))
        n = material.n
        p = math.sqrt(f * n * (f * n - thickness)) * (n - 1) / n
        R = f * (n - 1) + p * np.sign(f)
        super().__init__(R, -R, diameter, thickness, position, material, label, primitive, smooth, u, v, s)


class PlanoConvexLens(ThickLens):
    def __init__(
        self,
        f: float,
        diameter: float,
        thickness: float,
        material: RefractiveMaterial,
        position: Vector = Vector(0, 0, 0),
        label: str = "lens",
        primitive: str = primitives.DEFAULT,
        smooth: bool = True,
        u: int = 24,
        v: int = 2,
        s: int = 24,
    ):
        R1 = f * (material.n - 1)
        R2 = math.inf
        if f < 0:
            R1, R2 = R2, R1
        super().__init__(R1, R2, diameter, thickness, position, material, label, primitive, smooth, u, v, s)


class PlanoConcaveLens(ThickLens):
    def __init__(
        self,
        f: float,
        diameter: float,
        thickness: float,
        material: RefractiveMaterial,
        position: Vector = Vector(0, 0, 0),
        label: str = "lens",
        primitive: str = primitives.DEFAULT,
        smooth: bool = True,
        u: int = 24,
        v: int = 2,
        s: int = 24,
    ):
        R1 = math.inf
        R2 = f * (material.n - 1)
        if f < 0:
            R1, R2 = R2, R1
        super().__init__(R1, R2, diameter, thickness, position, material, label, primitive, smooth, u, v, s)
