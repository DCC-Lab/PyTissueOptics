from dataclasses import dataclass
from typing import List
from pytissueoptics.scene.geometry import BoundingBox, Polygon


@dataclass
class SplitNodeResult:
    stopCondition: bool
    boundingBoxes: List[BoundingBox]
    polygonGroups: List[List[Polygon]]
