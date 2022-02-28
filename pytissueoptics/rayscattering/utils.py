import math

from pytissueoptics.scene import Vector


def rotateVectorAround(vector: Vector, direction: Vector, theta: float) -> Vector:
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
    #
    # Expects `direction` to be normalized

    cost = math.cos(theta)
    sint = math.sin(theta)
    one_cost = 1 - cost

    ux = direction.x
    uy = direction.y
    uz = direction.z

    X = vector.x
    Y = vector.y
    Z = vector.z

    x = (cost + ux * ux * one_cost) * X \
        + (ux * uy * one_cost - uz * sint) * Y \
        + (ux * uz * one_cost + uy * sint) * Z
    y = (uy * ux * one_cost + uz * sint) * X \
        + (cost + uy * uy * one_cost) * Y \
        + (uy * uz * one_cost - ux * sint) * Z
    z = (uz * ux * one_cost - uy * sint) * X \
        + (uz * uy * one_cost + ux * sint) * Y \
        + (cost + uz * uz * one_cost) * Z
    return Vector(x, y, z)
