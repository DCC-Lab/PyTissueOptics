from typing import Tuple, List

from pytissueoptics.scene.geometry import Polygon


class PolygonCounter:
    def count(self, line: float, nodeAxis: str, polygons: List[Polygon]) -> List:
        raise NotImplementedError
