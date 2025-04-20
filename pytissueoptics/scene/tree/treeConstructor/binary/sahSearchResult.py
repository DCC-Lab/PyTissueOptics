from dataclasses import dataclass
from typing import List

from pytissueoptics.scene.geometry import BoundingBox, Polygon


@dataclass
class SAHSearchResult:
    leftPolygons: List[Polygon]
    rightPolygons: List[Polygon]
    splitPolygons: List[Polygon]
    leftBbox: BoundingBox
    rightBbox: BoundingBox
    splitAxis: str
    splitValue: float

    @property
    def SAH(self):
        return self.leftSAH + self.rightSAH

    @property
    def rightSAH(self):
        return self.nRight * self.rightBbox.getArea()

    @property
    def leftSAH(self):
        return self.nLeft * self.leftBbox.getArea()

    @property
    def nLeft(self):
        return len(self.leftPolygons)

    @property
    def nRight(self):
        return len(self.rightPolygons)
