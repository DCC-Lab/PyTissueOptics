from typing import Tuple, List
from pytissueoptics.scene.geometry import Polygon

class PolyCounter:
    def run(self, line: float, nodeAxis: str, polygons: List[Polygon]) -> Tuple:
        raise NotImplementedError


class BBoxPolyCounter(PolyCounter):
    def run(self, line: float, nodeAxis: str, polygons: List[Polygon]) -> Tuple:
        goingLeft = []
        goingRight = []

        for polygon in polygons:
            if polygon.bbox.getAxisLimit(nodeAxis, "min") < line and \
                    polygon.bbox.getAxisLimit(nodeAxis, "max") < line:
                goingLeft.append(polygon)
            elif polygon.bbox.getAxisLimit(nodeAxis, "min") > line and \
                    polygon.bbox.getAxisLimit(nodeAxis, "max") > line:
                goingRight.append(polygon)
            else:
                goingLeft.append(polygon)
                goingRight.append(polygon)

        return goingLeft, goingRight


class CentroidPolyCounter(PolyCounter):
    def run(self, line: float, nodeAxis: str, polygons: List[Polygon]) -> Tuple:
        goingLeft = []
        goingRight = []

        for polygon in polygons:
            if polygon.bbox.getAxisLimit(nodeAxis, "min") < line and \
                    polygon.bbox.getAxisLimit(nodeAxis, "max") < line:
                goingLeft.append(polygon)
            elif polygon.bbox.getAxisLimit(nodeAxis, "min") > line and \
                    polygon.bbox.getAxisLimit(nodeAxis, "max") > line:
                goingRight.append(polygon)
            else:
                goingLeft.append(polygon)
                goingRight.append(polygon)

        return goingLeft, goingRight