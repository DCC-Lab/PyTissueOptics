from typing import List

from pytissueoptics.scene.geometry import Polygon, BoundingBox


class Node:
    def __init__(self, parent: 'Node' = None, polygons: List[Polygon] = None, bbox: BoundingBox = None, depth: int = 0):

        self._parent = parent
        self._children = []
        self._polygons = polygons
        self._bbox = bbox
        self._depth = depth
        self._visited = False

    @property
    def children(self) -> List['Node']:
        return self._children

    @property
    def isRoot(self) -> bool:
        if self._parent is None:
            return True
        else:
            return False

    @property
    def isLeaf(self) -> bool:
        if not self._children:
            return True
        else:
            return False

    @property
    def polygons(self) -> List[Polygon]:
        return self._polygons

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def bbox(self) -> BoundingBox:
        return self._bbox
