from typing import List

from pytissueoptics.scene.geometry import BoundingBox, Vector
from pytissueoptics.scene.tree import TreeConstructor
from pytissueoptics.scene.tree import Node
from pytissueoptics.scene.solids import Cuboid
from pytissueoptics.scene.scene import Scene


class Tree:
    def __init__(self, scene: Scene, constructor: TreeConstructor, maxDepth=6, maxLeafSize=2):
        self._scene = scene
        self._maxDepth = maxDepth
        self._maxLeafSize = maxLeafSize
        self._polygons = self._scene.getPolygons()
        self._bbox = self._scene.getBoundingBox()
        self._constructor = constructor
        self._root = Node(polygons=self._polygons, bbox=self._bbox, maxDepth=maxDepth, maxLeafSize=maxLeafSize)
        self._constructor.growTree(self._root)

    def searchPoint(self, point: Vector) -> BoundingBox:
        raise NotImplementedError

    def searchRayIntersection(self, ray):
        raise NotImplementedError

    @property
    def maxDepth(self):
        return self._maxDepth

    @property
    def maxLeafSize(self):
        return self._maxLeafSize

    def getNodeCount(self, node=None):
        if node is None:
            node = self._root
        counter = 1
        for childNode in node.children:
            counter += self.getNodeCount(childNode)

        return counter

    def getLeafCount(self, node=None):
        if node is None:
            node = self._root
        counter = 0
        if node.isLeaf:
            counter += 1
        else:
            for childNode in node.children:
                counter += self.getLeafCount(childNode)
        return counter

    def getLeafNodes(self, node=None, nodesList=None):
        if nodesList is None and node is None:
            nodesList = []
            node = self._root

        if not node.isLeaf:
            for childNode in node.children:
                self.getLeafNodes(childNode, nodesList)

        else:
            nodesList.append(node)

        if node.isRoot:
            return nodesList

    def getLeafBoundingBoxes(self, node: Node = None, bboxList: List = None) -> List[BoundingBox]:
        if bboxList is None and node is None:
            bboxList = []
            node = self._root

        if not node.isLeaf:
            for childNode in node.children:
                self.getLeafBoundingBoxes(childNode, bboxList)

        else:
            bboxList.append(node.bbox)

        if node.isRoot:
            return bboxList

    def getLeafBoundingBoxesAsCuboids(self) -> List[Cuboid]:
        cuboids = []
        for bbox in self.getLeafBoundingBoxes():
            a = bbox.xMax - bbox.xMin
            b = bbox.yMax - bbox.yMin
            c = bbox.zMax - bbox.zMin
            cuboids.append(Cuboid(a=a, b=b, c=c, position=bbox.center))
        return cuboids

    def printBranching(self, node=None):
        if node is None:
            node = self._root
        if node.isLeaf:
            print(f"{node.id}")

        else:
            print(f"{node.id}")
            for child in node.children:
                self.printBranching(child)
