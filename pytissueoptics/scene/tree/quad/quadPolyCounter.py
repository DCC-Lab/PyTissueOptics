from typing import Tuple, List
from pytissueoptics.scene.geometry import Polygon

class QuadPolyCounter:
    def run(self, line: float, nodeAxis: str, polygons: List[Polygon]) -> Tuple:
        raise NotImplementedError
