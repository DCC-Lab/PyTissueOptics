from typing import List
from pytissueoptics.scene.geometry import Polygon, BoundingBox


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


class LargestSpanAxis(AxisSelector):
    def _run(self):
        bbox: BoundingBox = None
        for polygon in self._polygons:
            if bbox is None:
                bbox = polygon.bbox.newFrom()
            else:
                bbox.extendTo(polygon.bbox)

        widths = [bbox.xWidth, bbox.yWidth, bbox.zWidth]
        axisIndex = widths.index(max(widths))
        axes = ["x", "y", "z"]
        return axes[axisIndex]


class LowestSpanAxis(AxisSelector):
    def _run(self):
        bbox: BoundingBox = None
        for polygon in self._polygons:
            if bbox is None:
                bbox = polygon.bbox.newFrom()
            else:
                bbox.extendTo(polygon.bbox)

        widths = [bbox.xWidth, bbox.yWidth, bbox.zWidth]
        axisIndex = widths.index(min(widths))
        axes = ["x", "y", "z"]
        return axes[axisIndex]
