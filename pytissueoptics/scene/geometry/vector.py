

class Vector:
    """
    Basic implementation of a mutable 3D Vector. It implements most of the basic vector operation.
    Mutability is necessary when working with shared object references for expected behavior.
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

    def __truediv__(self, scalar: float) -> 'Vector':
        return Vector(self._x / scalar, self._y / scalar, self._z / scalar)

    def add(self, other: 'Vector'):
        self._x += other.x
        self._y += other.y
        self._z += other.z

    def subtract(self, other: 'Vector'):
        self._x -= other.x
        self._y -= other.y
        self._z -= other.z

    def multiply(self, scalar: float):
        self._x *= scalar
        self._y *= scalar
        self._z *= scalar

    def divide(self, scalar: float):
        self._x /= scalar
        self._y /= scalar
        self._z /= scalar

    def getNorm(self) -> float:
        return (self._x ** 2 + self._y ** 2 + self._z ** 2) ** (1 / 2)

    def normalize(self):
        norm = self.getNorm()
        if norm != 0:
            self._x = self._x / norm
            self._y = self._y / norm
            self._z = self._z / norm

    def cross(self, other: 'Vector') -> 'Vector':
        ux, uy, uz = self._x, self._y, self._z
        vx, vy, vz = other.x, other.y, other.z
        return Vector(uy*vz - uz*vy, uz*vx - ux*vz, ux*vy - uy*vx)

    @property
    def array(self) -> list:
        return [self._x, self._y, self._z]

    def update(self, x: float, y: float, z: float):
        self._x = x
        self._y = y
        self._z = z

    def copy(self) -> 'Vector':
        return Vector(self._x, self._y, self._z)
