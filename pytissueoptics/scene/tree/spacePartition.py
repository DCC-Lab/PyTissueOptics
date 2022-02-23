from typing import List

from pytissueoptics.scene.geometry import BoundingBox, Vector, Polygon
from pytissueoptics.scene.tree import TreeConstructor
from pytissueoptics.scene.tree import Node
from pytissueoptics.scene.solids import Cuboid


class SpacePartition:
    """
    This is a SpacePartition that saves the subdivion of the available space. The space it partitioned as a tree.
    The tree has nodes that split and contain other node, up to the leaf node, the smallest units.
    Each node has a boundingBox and a List of polygons that it contains.

    To construct a SpacePartition, you must give those parameters.
    - bbox : The space of interest that will be subdivided.
    - polygons: The List[Polygon] from which the subdivision will make its choices.
    - constructor: A TreeConstructor class which will define the subdivision behaviour
    - maxDepth: the maximum depth of a node, will limit the creation of node to a certain level
    - minLeafSize: the minimum amount of polygons a leaf can contain. No leaf node can have less polygons.
    """

    def __init__(self, bbox: BoundingBox, polygons: List[Polygon], constructor: TreeConstructor, maxDepth=6,
                 minLeafSize=2):
        self._maxDepth = maxDepth
        self._maxLeafSize = minLeafSize
        self._polygons = polygons
        self._bbox = bbox
        self._constructor = constructor
        self._root = Node(polygons=self._polygons, bbox=self._bbox)
        self._constructor.growTree(self._root, maxDepth=maxDepth, maxLeafSize=minLeafSize)

    def searchPoint(self, point: Vector, node: Node = None) -> Node:
        if node is None:
            node = self._root

        if node.isLeaf:
            return node

        isInside = None
        for child in node.children:
            if child.bbox.contains(point):
                isInside = child
                break

        self.searchPoint(point, isInside)

    @property
    def maxDepth(self):
        return self._maxDepth

    @property
    def minLeafSize(self):
        return self._minLeafSize

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

    def getLeafBoundingBoxes(self) -> List[BoundingBox]:
        nodesList = self.getLeafNodes()
        nodesBbox = [node.bbox for node in nodesList]
        return nodesBbox

    def getLeafBoundingBoxesAsCuboids(self) -> List[Cuboid]:
        cuboids = []
        for bbox in self.getLeafBoundingBoxes():
            a = bbox.xMax - bbox.xMin
            b = bbox.yMax - bbox.yMin
            c = bbox.zMax - bbox.zMin
            cuboids.append(Cuboid(a=a, b=b, c=c, position=bbox.center))
        return cuboids
