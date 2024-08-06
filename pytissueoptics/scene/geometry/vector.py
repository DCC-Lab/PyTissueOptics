import math


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

    @x.setter
    def x(self, value):
        self._x = value

    @y.setter
    def y(self, value):
        self._y = value

    @z.setter
    def z(self, value):
        self._z = value

    def __repr__(self):
        return f"<Vector>:({self.x}, {self.y}, {self.z})"

    def __eq__(self, other: 'Vector'):
        tol = 1e-5
        if math.isclose(other._x, self.x, abs_tol=tol) and math.isclose(other._y, self.y, abs_tol=tol) and math.isclose(other._z, self.z, abs_tol=tol):
            return True
        else:
            return False

    def __sub__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x - other.x, self.y - other._y, self.z - other._z)

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x + other.x, self.y + other._y, self.z + other._z)

    def __mul__(self, scalar: float) -> 'Vector':
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> 'Vector':
        return Vector(self.x / scalar, self.y / scalar, self.z / scalar)

    def add(self, other: 'Vector'):
        self.x += other._x
        self.y += other._y
        self.z += other._z

    def subtract(self, other: 'Vector'):
        self.x -= other._x
        self.y -= other._y
        self.z -= other._z

    def multiply(self, scalar: float):
        self.x *= scalar
        self.y *= scalar
        self.z *= scalar

    def divide(self, scalar: float):
        self.x /= scalar
        self.y /= scalar
        self.z /= scalar

    def getNorm(self) -> float:
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** (1 / 2)

    def normalize(self):
        norm = self.getNorm()
        if norm != 0:
            self.x = self.x / norm
            self.y = self.y / norm
            self.z = self.z / norm

    def cross(self, other: 'Vector') -> 'Vector':
        ux, uy, uz = self.x, self.y, self.z
        vx, vy, vz = other._x, other._y, other._z
        return Vector(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)

    def dot(self, other: 'Vector') -> float:
        return self.x*other._x + self.y*other._y + self.z*other._z

    @property
    def array(self) -> list:
        return [self.x, self.y, self.z]

    def update(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def copy(self) -> 'Vector':
        return Vector(self.x, self.y, self.z)

    def rotateAround(self, unitAxis: 'Vector', theta: float):
        """
        Rotate the vector around `unitAxis` by `theta` radians. Assumes the axis to be a unit vector.
        """
        # This is the most expensive (and most common)
        # operation when performing Monte Carlo in tissue
        # (15% of time spent here). It is difficult to optimize without
        # making it even less readable than it currently is
        # http://en.wikipedia.org/wiki/Rotation_matrix
        #
        # Several options were tried in the past such as
        # external not-so-portable C library, unreadable
        # shortcuts, sine and cosine lookup tables, etc...
        # and the performance gain was minimal (<20%).
        # For now, this is the best, most readable solution.

        cost = math.cos(theta)
        sint = math.sin(theta)
        one_cost = 1 - cost

        ux = unitAxis.x
        uy = unitAxis.y
        uz = unitAxis.z

        X = self.x
        Y = self.y
        Z = self.z

        x = (cost + ux * ux * one_cost) * X \
            + (ux * uy * one_cost - uz * sint) * Y \
            + (ux * uz * one_cost + uy * sint) * Z
        y = (uy * ux * one_cost + uz * sint) * X \
            + (cost + uy * uy * one_cost) * Y \
            + (uy * uz * one_cost - ux * sint) * Z
        z = (uz * ux * one_cost - uy * sint) * X \
            + (uz * uy * one_cost + ux * sint) * Y \
            + (cost + uz * uz * one_cost) * Z

        self.update(x, y, z)
        self.normalize()

    def getAnyOrthogonal(self) -> 'Vector':
        if abs(self.z) < abs(self.x):
            return Vector(self.y, -self.x, 0)

        return Vector(0, -self.z, self.y)

    def __hash__(self):
        return hash((self.x, self.y, self.z))
