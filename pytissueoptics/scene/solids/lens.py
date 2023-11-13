import math
from typing import List

import numpy as np

from pytissueoptics.scene.geometry import Vector, primitives, Vertex
from pytissueoptics.scene.solids import Cylinder


class Lens(Cylinder):
    """
    The Lens is defined by a diameter, a thickness, a front radius and a back radius.
    The position refers to the vector from global origin to its centroid.
    The generated mesh will be divided into the following subgroups: Front, Side, Back.
    By default, front will point towards the negative z-axis.
    """
    def __init__(self, frontRadius: float, backRadius: float, diameter: float = 2.54, thickness: float = 0,
                 position: Vector = Vector(0, 0, 0), material=None, u: int = 32, v: int = 2, s: int = 10,
                 label: str = "lens", primitive: str = primitives.DEFAULT, smooth: bool = True):

        self._diameter = diameter
        self._frontRadius = frontRadius
        self._backRadius = backRadius

        super().__init__(radius=diameter / 2, length=thickness, u=u, v=v, s=s, position=position,
                         material=material, label=label, primitive=primitive, smooth=smooth)

    def _computeVerticesOfLayers(self) -> tuple:
        frontLayers, lateralLayers, backLayers = super()._computeVerticesOfLayers()
        frontVertices = self._vertices[:len(frontLayers) * self._u] + [self._frontCenter]
        self._applyCurvature(self._frontRadius, frontVertices)
        backVertices = self._vertices[-len(backLayers) * self._u:] + [self._backCenter]
        self._applyCurvature(self._backRadius, backVertices)
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

    def smooth(self, surfaceLabel: str = None):
        if surfaceLabel:
            return super(Cylinder, self).smooth(surfaceLabel)
        for surfaceLabel in self._surfaces.surfaceLabels:
            self.smooth(surfaceLabel)
