from typing import List
from pytissueoptics.scene.geometry import Polygon, BoundingBox


class AxisSelector:
    def run(self, depth: int, nodeBbox: BoundingBox, polygons: List[Polygon]) -> str:
        raise NotImplementedError
