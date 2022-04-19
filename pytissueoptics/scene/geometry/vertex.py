from pytissueoptics.scene.geometry import Vector


class Vertex(Vector):
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        super().__init__(x, y, z)
        self.normal = None
