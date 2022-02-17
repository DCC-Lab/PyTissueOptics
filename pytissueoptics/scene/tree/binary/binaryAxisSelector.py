from typing import List
from pytissueoptics.scene.geometry import BoundingBox
from pytissueoptics.scene.tree.splitUtils import AxisSelector


class BinaryAxisSelector(AxisSelector):
    def _run(self) -> List[str]:
        raise NotImplementedError


class RotateAxis(BinaryAxisSelector):
    def _run(self) -> List[str]:
        axis = self._depth % 3
        if axis == 0:
            return ["x"]
        elif axis == 1:
            return ["y"]
        elif axis == 2:
            return ["z"]


class LargestSpanAxis(AxisSelector):
    def _run(self) -> List[str]:
        bbox: BoundingBox = None
        for polygon in self._polygons:
            if bbox is None:
                bbox = polygon.bbox.newFrom()
            else:
                bbox.extendTo(polygon.bbox)

        widths = [bbox.xWidth, bbox.yWidth, bbox.zWidth]
        axisIndex = widths.index(max(widths))
        axes = ["x", "y", "z"]
        return [axes[axisIndex]]
