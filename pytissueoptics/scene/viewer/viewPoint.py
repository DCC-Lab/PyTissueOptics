from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


@dataclass
class ViewPoint:
    azimuth: Optional[float]
    elevation: Optional[float]
    distance: Optional[float]
    focalpoint: Optional[np.ndarray]
    roll: Optional[float]


class ViewPointStyle(Enum):
    OPTICS = 0
    NATURAL = 1
    NATURAL_FRONT = 2


class ViewPointFactory:
    def create(self, viewPointStyle: ViewPointStyle):
        if viewPointStyle == ViewPointStyle.OPTICS:
            return self.getOpticsViewPoint()
        elif viewPointStyle == ViewPointStyle.NATURAL:
            return self.getNaturalViewPoint()
        elif viewPointStyle == ViewPointStyle.NATURAL_FRONT:
            return self.getNaturalFrontViewPoint()
        else:
            raise ValueError(f"Invalid viewpoint style: {viewPointStyle}")

    @staticmethod
    def getOpticsViewPoint():
        return ViewPoint(azimuth=-30, elevation=215, distance=None, focalpoint=None, roll=0)

    @staticmethod
    def getNaturalViewPoint():
        return ViewPoint(azimuth=30, elevation=30, distance=None, focalpoint=None, roll=0)

    @staticmethod
    def getNaturalFrontViewPoint():
        return ViewPoint(azimuth=0, elevation=0, distance=None, focalpoint=None, roll=None)
