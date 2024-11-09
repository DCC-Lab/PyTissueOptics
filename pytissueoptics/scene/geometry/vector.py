import math

COS_ZERO = 1 - 1e-12


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
        return f"<Vector>:({self._x}, {self._y}, {self._z})"

    def __eq__(self, other: 'Vector'):
        tol = 1e-5
        if math.isclose(other._x, self._x, abs_tol=tol) and math.isclose(other._y, self._y, abs_tol=tol) and math.isclose(other._z, self._z, abs_tol=tol):
            return True
        else:
            return False

    def __sub__(self, other: 'Vector') -> 'Vector':
        return Vector(self._x - other._x, self._y - other._y, self._z - other._z)

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self._x + other._x, self._y + other._y, self._z + other._z)

    def __mul__(self, scalar: float) -> 'Vector':
        return Vector(self._x * scalar, self._y * scalar, self._z * scalar)

    def __truediv__(self, scalar: float) -> 'Vector':
        return Vector(self._x / scalar, self._y / scalar, self._z / scalar)

    def add(self, other: 'Vector'):
        self._x += other._x
        self._y += other._y
        self._z += other._z

    def subtract(self, other: 'Vector'):
        self._x -= other._x
        self._y -= other._y
        self._z -= other._z

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
        vx, vy, vz = other._x, other._y, other._z
        return Vector(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)

    def dot(self, other: 'Vector') -> float:
        return self._x*other._x + self._y*other._y + self._z*other._z

    @property
    def array(self) -> list:
        return [self._x, self._y, self._z]

    def update(self, x: float, y: float, z: float):
        self._x = x
        self._y = y
        self._z = z

    def copy(self) -> 'Vector':
        return Vector(self._x, self._y, self._z)

    def spin(self, theta: float, phi: float):
        """ In spherical coordinates. Taken directly from MCML code. """
        cosp = math.cos(phi)
        if phi < math.pi:
            sinp = math.sqrt(1 - cosp*cosp)
        else:
            sinp = -math.sqrt(1 - cosp*cosp)

        cost = math.cos(theta)
        sint = math.sqrt(1 - cost*cost)

        if abs(self._z) > COS_ZERO:
            ux = sint*cosp
            uy = sint*sinp
            uz = cost * (1 if self._z >= 0 else -1)
        else:
            temp = math.sqrt(1 - self._z*self._z)
            ux = sint*(self._x*self._z*cosp - self._y*sinp) / temp + self._x*cost
            uy = sint*(self._y*self._z*cosp + self._x*sinp) / temp + self._y*cost
            uz = -sint*cosp*temp + self._z*cost

        self.update(ux, uy, uz)

    def rotateAround(self, unitAxis: 'Vector', theta: float):
        """
        Rotate the vector around `unitAxis` by `theta` radians. Assumes the axis to be a unit vector.
        Uses Rodrigues' rotation formula.
        """
        cost = math.cos(theta)
        sint = math.sin(theta)
        one_cost = 1 - cost

        ux = unitAxis.x
        uy = unitAxis.y
        uz = unitAxis.z

        X = self._x
        Y = self._y
        Z = self._z

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

    def getAnyOrthogonal(self) -> 'Vector':
        if abs(self._z) < abs(self._x):
            return Vector(self._y, -self._x, 0)

        return Vector(0, -self._z, self._y)

    def __hash__(self):
        return hash((self._x, self._y, self._z))
