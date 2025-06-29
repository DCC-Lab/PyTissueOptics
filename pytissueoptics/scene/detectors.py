import math
from abc import ABC

from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.solids.circle import Circle
from pytissueoptics.scene.solids.rectangle import Rectangle


class Detector(ABC):
    """
    The light is only collected when going against the specified orientation of the surface, and with an angle of incidence
    within the numerical aperture.
    """
    def __init__(self, solid: Solid, halfAngle: float = math.pi / 2):
        if not isinstance(solid, Circle) and not isinstance(solid, Rectangle):
            # TODO: Ignore numerical aperture if vertices don't have the same direction.
            raise NotImplementedError("Detector geometry must be a Circle or a Rectangle.")

        self._solid = solid
        self._halfAngle = halfAngle
        self._solid.setOutsideEnvironment(self._solid.getEnvironment())
        self._solid.setDetectionHalfAngle(halfAngle)

    @property
    def solid(self) -> Solid:
        return self._solid


class CircleDetector(Detector):
    def __init__(self, radius: float, orientation: Vector = Vector(0, 0, 1), position: Vector = Vector(0, 0, 0),
                 halfAngle: float = math.pi / 2, label="circleDetector"):
        super().__init__(Circle(radius, orientation, position, label=label), halfAngle=halfAngle)


class RectangleDetector(Detector):
    """
    Parameters a and b are the rectangle dimensions along the x and y axes, respectively, when pointing towards the z axis.
    """
    def __init__(self, a: float, b: float, orientation=Vector(0, 0, 1), position=Vector(0, 0, 0),
                 halfAngle: float = math.pi / 2, label="rectangleDetector"):
        super().__init__(solid=Rectangle(a, b, orientation, position, label=label), halfAngle=halfAngle)
