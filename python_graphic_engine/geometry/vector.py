

class Vector:
    """
    Basic implementation of a 3D Vector. It implements most of the basic vector operation.
    """

    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self._x = x
        self._y = y
        self._z = z

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    def __repr__(self):
        return f"<Vector>:({self.x}, {self.y}, {self.z})"

    def __eq__(self, other: 'Vector'):
        if other.x == self.x and other.y == self.y and other.z == self.z:
            return True
        else:
            return False

    def __sub__(self, other: 'Vector') -> 'Vector':
        return Vector(self._x - other.x, self._y - other.y, self._z - other.z)

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self._x + other.x, self._y + other.y, self._z + other.z)

    def norm(self) -> float:
        return (self._x ** 2 + self._y ** 2 + self._z ** 2) ** (1 / 2)

    def normalize(self):
        norm = self.norm()
        self._x = self._x / norm
        self._y = self._y / norm
        self._z = self._z / norm
