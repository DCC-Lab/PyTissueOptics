import numpy as np

class Vector:
    npStruct = np.dtype([("x", np.float32),("y", np.float32),("z", np.float32)])

    """
    Basic implementation of a mutable 3D Vector. It implements most of the basic vector operation.
    Mutability is necessary when working with shared object references for expected behavior.
    Internally, the (x,y,z) components are kept in a numpy.array.  This facilitates computation
    but it also allows to provide our own array that can be used.  This is the central part
    of the CompactVector class that uses an already allocated structured numpy array to keep 
    many vectors together in a large array.
    """

    def __init__(self, x: float = 0, y: float = 0, z: float = 0, array=None):
        if array is not None:
            self._data = array
        else:
            self._data = np.array([x,y,z], dtype=np.float32)

    @property
    def array(self) -> np.array:
        return self._data

    @array.setter
    def array(self, new_values):
        self._data[0] = new_values.x
        self._data[1] = new_values.y
        self._data[2] = new_values.z

    @property
    def x(self):
        return self._data[0]

    @property
    def y(self):
        return self._data[1]

    @property
    def z(self):
        return self._data[2]

    @x.setter
    def x(self, value):
        self._data[0] = value

    @y.setter
    def y(self, value):
        self._data[1] = value

    @z.setter
    def z(self, value):
        self._data[2] = value

    def __repr__(self):
        return f"<Vector>:({self.x}, {self.y}, {self.z})"

    def __eq__(self, other: 'Vector'):
        tol = 1e-5
        if np.isclose(other.x, self.x, atol=tol) and np.isclose(other.y, self.y, atol=tol) and np.isclose(other.z, self.z, atol=tol):
            return True
        else:
            return False

    def __sub__(self, other: 'Vector') -> 'Vector':
        # return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
        return Vector(array=self._data-other._data)

    def __add__(self, other: 'Vector') -> 'Vector':
        # return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
        return Vector(array=self._data+other._data)

    def __mul__(self, scalar: float) -> 'Vector':
        # return Vector(self.x * scalar, self.y * scalar, self.z * scalar)
        return Vector(array=self._data * scalar)

    def __truediv__(self, scalar: float) -> 'Vector':
        # return Vector(self.x / scalar, self.y / scalar, self.z / scalar)
        return Vector(array=self._data / scalar)

    def add(self, other: 'Vector'):
        self._data += other._data

    def subtract(self, other: 'Vector'):
        self._data -= other._data

    def multiply(self, scalar: float):
        self._data *= scalar

    def divide(self, scalar: float):
        self._data /= scalar

    def getNorm(self) -> float:
        return np.linalg.norm(self._data)

    def normalize(self):
        norm = self.getNorm()
        self._data /= norm

    def cross(self, other: 'Vector') -> 'Vector':
        ux, uy, uz = self.x, self.y, self.z
        vx, vy, vz = other.x, other.y, other.z
        return Vector(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)

    def dot(self, other: 'Vector') -> float:
        return self._data@other._data

    def update(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def copy(self) -> 'Vector':
        return Vector(self.x, self.y, self.z)

    # def rotateAround(self, unitAxis: 'Vector', theta: float):
    #     self.rotateAround_Matrix(unitAxis, theta)

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

        cost = np.cos(theta)
        sint = np.sin(theta)
        one_cost = 1 - cost

        ux = unitAxis.x
        uy = unitAxis.y
        uz = unitAxis.z

        uxuy=(ux * uy)
        uxuz=(ux * uz)
        uyuz=(uy * uz)

        rot = np.array([[(cost + ux * ux * one_cost), (uxuy * one_cost - uz * sint),(uxuz * one_cost + uy * sint)],
                        [(uxuy * one_cost + uz * sint),(cost + uy * uy * one_cost),(uyuz * one_cost - ux * sint)],
                        [(uxuz * one_cost - uy * sint), (uyuz * one_cost + ux * sint),  (cost + uz * uz * one_cost) ]
                       ])

        result = rot@self._data
        self.update(result[0], result[1], result[2])

    def rotateAround_Rodrigue(self, unitAxis: 'Vector', theta: float):
        """
        Rotate the vector around `unitAxis` by `theta` radians. Assumes the axis to be a unit vector.
        It could have been faster, but it is not. Leaving here for historical reasons.
        """

        cost = np.cos(theta)
        sint = np.sin(theta)
        one_cost = 1 - cost

        # # Rodrigues' rotation formula
        result = self * cost + unitAxis.cross(self) * sint + unitAxis * unitAxis.dot(self) * one_cost

        self.update(result.x, result.y, result.z)

    def getAnyOrthogonal(self) -> 'Vector':
        if abs(self.z) < abs(self.x):
            return Vector(self.y, -self.x, 0)

        return Vector(0, -self.z, self.y)

    def __hash__(self):
        return hash((self.x, self.y, self.z))


class CompactVector(Vector):
    def __init__(self, rawBuffer, index=0, offset=0, stride=0):
        super().__init__(array=np.frombuffer(rawBuffer, dtype=np.float32, count=3, offset=offset+index*stride))
