from pytissueoptics.scene.geometry import Polygon, Vector, Environment


class Triangle(Polygon):
    def __init__(self, v1: Vector, v2: Vector, v3: Vector,
                 insideEnvironment: Environment = None, outsideEnvironment: Environment = None, normal: Vector = None):
        super().__init__(vertices=[v1, v2, v3],
                         insideEnvironment=insideEnvironment, outsideEnvironment=outsideEnvironment, normal=normal)
