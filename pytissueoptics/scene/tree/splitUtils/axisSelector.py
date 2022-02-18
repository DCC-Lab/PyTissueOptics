from typing import List
from pytissueoptics.scene.geometry import Polygon, BoundingBox


class AxisSelector:
    def __init__(self):
        self._depth = None
        self._polygons = None
        self.nodeBbox = None

    def run(self, depth: int, nodeBbox: BoundingBox, polygons: List[Polygon]) -> List[str]:
        self._depth = depth
        self._polygons = polygons
        self.nodeBbox = nodeBbox
        return self._run()

    def _run(self) -> List[str]:
        raise NotImplementedError
