from typing import Tuple, List

from pytissueoptics import Intersection


class Logger:
    def __init__(self):
        self._positions = []
        self._energy = []
        self._intersections = []

    def logEnergy(self, positions: list, energy: list):
        self._positions.extend(positions)
        self._energy.extend(energy)

    def logIntersections(self, intersections: List[Intersection]):
        self._intersections.extend(intersections)

    def getEnergy(self) -> Tuple[list, list]:
        return self._positions, self._energy

    def getIntersections(self) -> List[Intersection]:
        return self._intersections
