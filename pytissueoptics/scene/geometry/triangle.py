from .polygon import Environment, Polygon
from .vector import Vector
from .vertex import Vertex


class Triangle(Polygon):
    def __init__(
        self,
        v1: Vertex,
        v2: Vertex,
        v3: Vertex,
        insideEnvironment: Environment = None,
        outsideEnvironment: Environment = None,
        normal: Vector = None,
    ):
        super().__init__(
            vertices=[v1, v2, v3],
            insideEnvironment=insideEnvironment,
            outsideEnvironment=outsideEnvironment,
            normal=normal,
        )
