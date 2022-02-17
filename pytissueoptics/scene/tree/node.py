from typing import List
from pytissueoptics.scene.geometry import Polygon, BoundingBox, Vector
from pytissueoptics.scene.tree import TreeStrategy
from pytissueoptics.scene.scene import Scene


class Node:
    def __init__(self, parent: 'Node' = None, children: List['Node'] = None, treeStrategy: TreeStrategy = None,
                 scene: Scene = None, polygons: List[Polygon] = None, bbox: BoundingBox = None, depth: int = 0,
                 maxDepth=100):

        self._parent = parent
        self._children = children
        self._scene = scene
        self._polygons = polygons
        self._bbox = bbox
        self._treeStrategy = treeStrategy
        self._depth = depth
        self._maxDepth = maxDepth

        if self.isRoot and scene is not None:
            self._polygons = self._scene.getPolygons()
            self._bbox = self._scene.getBoundingBox()

        self.split()

    @property
    def parent(self):
        return self._parent

    @property
    def isRoot(self):
        if self._parent is None:
            return True
        else:
            return False

    @property
    def isLeaf(self):
        if not self._children:
            return True
        else:
            return False

    @property
    def children(self):
        return self._children

    @property
    def polygons(self) -> List[Polygon]:
        return self._polygons

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def bbox(self) -> BoundingBox:
        return self._bbox

    def split(self):
        if self._depth < self._maxDepth and len(self._polygons) > 2:
            splitNodeResult = self._split()
            if not splitNodeResult.stopCondition:
                for i, polygonGroup in enumerate(splitNodeResult.polygonGroups):
                    if len(polygonGroup) > 0:
                        childNode = Node(parent=self, polygons=polygonGroup,
                                         bbox=splitNodeResult.bboxes[i], depth=self._depth + 1,
                                         maxDepth=self._maxDepth, treeStrategy=self._treeStrategy)
                        self._children.append(childNode)

    def _split(self):
        return self._treeStrategy.run(self)

    def searchPoint(self, point: Vector):
        raise NotImplementedError

    def searchRayIntersection(self, ray):
        raise NotImplementedError

    def getLeafBoundingBoxes(self, node: 'Node', bboxList: List) -> List[BoundingBox]:
        if bboxList is None:
            bboxList = []

        if node is not None:
            if not node.isLeaf:
                for childNode in self._children:
                    node.getLeafBoundingBoxes(childNode, bboxList)

            else:
                bboxList.append(node.boundingBox)

            if node.isRoot:
                return bboxList
