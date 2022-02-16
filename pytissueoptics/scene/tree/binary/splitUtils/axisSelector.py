from typing import Tuple, List
from pytissueoptics.scene.geometry import Polygon


class AxisSelector:
    def __init__(self):
        self._nodeAxis = None
        self._polygons = None

    def run(self, nodeAxis: str, polygons: List[Polygon]) -> str:
        self._nodeAxis = nodeAxis
        self._polygons = polygons
        return self._run()

    def _run(self) -> str:
        raise NotImplementedError


class RotateAxis(AxisSelector):
    def _run(self) -> str:
        if self._nodeAxis == "x":
            return "y"
        elif self._nodeAxis == "y":
            return "z"
        elif self._nodeAxis == "z":
            return "x"
