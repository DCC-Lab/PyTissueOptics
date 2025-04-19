from pytissueoptics.scene.geometry import Environment, Polygon, Vector, Vertex


class Quad(Polygon):
    def __init__(self, v1: Vertex, v2: Vertex, v3: Vertex, v4: Vertex,
                 insideEnvironment: Environment = None, outsideEnvironment: Environment = None, normal: Vector = None):
        super().__init__(vertices=[v1, v2, v3, v4],
                         insideEnvironment=insideEnvironment, outsideEnvironment=outsideEnvironment, normal=normal)
