import numpy as np
from pytissueoptics.scene.geometry.vector import Vector


class Rotation:
    def __init__(self, xTheta: float = 0, yTheta: float = 0, zTheta: float = 0):
        self._xTheta = xTheta
        self._yTheta = yTheta
        self._zTheta = zTheta

    @property
    def xTheta(self):
        return self._xTheta

    @property
    def yTheta(self):
        return self._yTheta

    @property
    def zTheta(self):
        return self._zTheta

    def add(self, other: 'Rotation'):
        self._xTheta += other.xTheta
        self._yTheta += other.yTheta
        self._zTheta += other.zTheta

    def __bool__(self):
        return self._xTheta != 0 or self._yTheta != 0 or self._zTheta != 0

    @classmethod
    def between(cls, fromDirection: Vector, toDirection: Vector):
        """ Create the euler rotation needed to go from one orientation to another. """
        fromDirection.normalize()
        toDirection.normalize()
        dot = fromDirection.dot(toDirection)
        dot = max(min(dot, 1), -1)
        angle = np.arccos(dot)
        axis = fromDirection.cross(toDirection)
        axis.normalize()
        axis = axis.array

        if np.isclose(angle, np.pi):
            if np.isclose(axis[0], 0):
                xTheta = np.degrees(np.pi)
                yTheta = 0
                zTheta = np.degrees(np.arctan2(-axis[1], axis[2]))
            else:
                xTheta = 0
                yTheta = np.degrees(np.pi)
                zTheta = np.degrees(np.arctan2(axis[0], axis[2]))
        else:
            xTheta = np.degrees(np.arctan2(axis[0]*np.sin(angle) - axis[2]*axis[1]*(1 - np.cos(angle)), 1 - (axis[0]**2 + axis[2]**2)*(1 - np.cos(angle))))
            zTheta = np.degrees(np.arctan2(axis[2]*np.sin(angle) - axis[1]*axis[0]*(1 - np.cos(angle)), 1 - (axis[2]**2 + axis[1]**2)*(1 - np.cos(angle))))
            yTheta = np.degrees(np.arctan2(axis[1]*np.sin(angle) - axis[0]*axis[2]*(1 - np.cos(angle)), 1 - (axis[1]**2 + axis[0]**2)*(1 - np.cos(angle))))
        return cls(xTheta, yTheta, zTheta)
