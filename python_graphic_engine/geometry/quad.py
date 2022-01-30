from graphics.geometry import Polygon, Vector


class Quad(Polygon):
    def __init__(self, v1: Vector, v2: Vector, v3: Vector, v4: Vector):
        super().__init__(vertices=[v1, v2, v3, v4])
