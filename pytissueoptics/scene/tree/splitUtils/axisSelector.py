from typing import List
from pytissueoptics.scene.geometry import Polygon, BoundingBox


class AxisSelector:
    def __init__(self):
        self._depth = None
        self._polygons = None

    def run(self, depth: int, polygons: List[Polygon]) -> List[str]:
        self._depth = depth
        self._polygons = polygons
        return self._run()

    def _run(self) -> List[str]:
        raise NotImplementedError
