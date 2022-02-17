from typing import Tuple, List
from pytissueoptics.scene.geometry import Polygon

class BinaryPolyCounter:
    def run(self, line: float, nodeAxis: str, polygons: List[Polygon]) -> Tuple:
        raise NotImplementedError


class BBoxPolyCounter(BinaryPolyCounter):
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


class CentroidPolyCounter(BinaryPolyCounter):
    def run(self, line: float, nodeAxis: str, polygons: List[Polygon]) -> Tuple:
        goingLeft = []
        goingRight = []
        centroidComponent = 0
        for polygon in polygons:
            if nodeAxis == "x":
                centroidComponent = polygon.centroid.x
            elif nodeAxis == "y":
                centroidComponent = polygon.centroid.y
            elif nodeAxis == "z":
                centroidComponent = polygon.centroid.z

            if centroidComponent < line:
                goingLeft.append(polygon)
            elif centroidComponent > line:
                goingRight.append(polygon)
            else:
                goingLeft.append(polygon)
                goingRight.append(polygon)

        return goingLeft, goingRight
