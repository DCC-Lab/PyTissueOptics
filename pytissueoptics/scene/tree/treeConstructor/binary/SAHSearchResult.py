from dataclasses import dataclass
from typing import List

from pytissueoptics.scene.geometry import Polygon, BoundingBox


@dataclass
class SAHSearchResult:
    lPolygons: List[Polygon]
    rPolygons: List[Polygon]
    mPolygons: List[Polygon]
    lBbox: BoundingBox
    rBbox: BoundingBox
    nLeft: int
    nRight: int
    lSAH: float
    rSAH: float
    totalSAH: float
    splitAxis: str
    splitValue: float
