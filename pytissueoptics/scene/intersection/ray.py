from pytissueoptics.scene.geometry import Vector


class Ray:
    def __init__(self, origin: Vector, direction: Vector):
        self._origin = origin
        self._direction = direction

    @property
    def origin(self):
        return self._origin

    @property
    def direction(self):
        return self._direction
