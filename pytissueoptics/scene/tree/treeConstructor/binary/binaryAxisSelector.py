from typing import List
from pytissueoptics.scene.geometry import BoundingBox, Polygon
from pytissueoptics.scene.tree.treeConstructor import AxisSelector


class RotateAxis(AxisSelector):
    def run(self, depth: int, nodeBbox: BoundingBox, polygons: List[Polygon]) -> str:
        axis = depth % 3
        if axis == 0:
            return "x"
        elif axis == 1:
            return "y"
        elif axis == 2:
            return "z"


class LargestPolygonSpanAxis(AxisSelector):
    def run(self, depth: int, nodeBbox: BoundingBox, polygons: List[Polygon]) -> str:
        bbox: BoundingBox = None
        for polygon in polygons:
            if bbox is None:
                bbox = polygon.bbox.copy()
            else:
                bbox.extendTo(polygon.bbox)

        widths = [bbox.xWidth, bbox.yWidth, bbox.zWidth]
        axisIndex = widths.index(max(widths))
        axes = ["x", "y", "z"]
        return axes[axisIndex]


class LargestSpanAxis(AxisSelector):
    def run(self, depth: int, nodeBbox: BoundingBox, polygons: List[Polygon]) -> str:
        bbox = nodeBbox
        widths = [bbox.xWidth, bbox.yWidth, bbox.zWidth]
        axisIndex = widths.index(max(widths))
        axes = ["x", "y", "z"]
        return axes[axisIndex]
