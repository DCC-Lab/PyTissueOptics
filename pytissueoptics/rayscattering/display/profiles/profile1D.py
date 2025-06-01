from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from pytissueoptics.rayscattering.display.utils import Direction
from pytissueoptics.rayscattering.energyLogging import EnergyType


class Profile1D:
    """
    Since 1D profiles are easily generated from existing 2D views or 3D data, this class is only used as a small
    dataclass. Only used internally Profile1DFactory when Viewer.show1D() is called. The user should only use the
    endpoint Viewer.show1D() which doesn't require creating a Profile1D object.
    """

    def __init__(
        self,
        data: np.ndarray,
        horizontalDirection: Direction,
        limits: Tuple[float, float],
        name: str = None,
        energyType=EnergyType.DEPOSITION,
    ):
        self.data = data
        self.limits = limits
        self.horizontalDirection = horizontalDirection
        self.name = name
        self.energyType = energyType

    def show(self, logScale: bool = True):
        limits = sorted(self.limits)
        if self.horizontalDirection.isNegative:
            self.data = np.flip(self.data, axis=0)
            limits = (limits[1], limits[0])
        bins = np.linspace(limits[0], limits[1], self.data.size + 1)[:-1]

        plt.bar(bins, self.data, width=np.diff(bins)[0], align="edge")

        if logScale:
            plt.yscale("log")
        plt.title(self.name)
        plt.xlim(*limits)
        plt.xlabel("xyz"[self.horizontalDirection.axis])
        plt.ylabel("Deposited energy" if self.energyType == EnergyType.DEPOSITION else "Fluence rate")
        plt.show()
