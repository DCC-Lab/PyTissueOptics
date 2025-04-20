from typing import List, Optional

from pytissueoptics.scene.geometry import BoundingBox, Polygon, Vector

from .node import Node
from .treeConstructor import TreeConstructor


class SpacePartition:
    """
    This is a SpacePartition that saves the subdivision of the available space. The space is partitioned as a tree.
    The tree has nodes that split and contain other nodes, up to the leaf node, the smallest units.
    Each node has a boundingBox and a List of polygons that it contains.

    To construct a SpacePartition, you must give those parameters.
    - bbox : The space of interest that will be subdivided.
    - polygons: The List[Polygon] from which the subdivision will make its choices.
    - constructor: A TreeConstructor class which will define the subdivision behaviour
    - maxDepth: the maximum depth of a node, will limit the creation of node to a certain level
    - minLeafSize: the minimum amount of polygons a leaf can contain. No leaf node can have fewer polygons.

    It is essentially a dataclass that contains a tree of nodes. The methods defined bellow are only used
    internally for benchmarking purposes.
    """

    def __init__(self, bbox: BoundingBox, polygons: List[Polygon], constructor: TreeConstructor, maxDepth=6,
                 minLeafSize=2):
        self._maxDepth = maxDepth
        self._minLeafSize = minLeafSize
        self._polygons = polygons
        self._bbox = bbox
        self._constructor = constructor
        self._root = Node(polygons=self._polygons, bbox=self._bbox)
        self._constructor.constructTree(self._root, maxDepth=maxDepth, minLeafSize=minLeafSize)

    @property
    def root(self) -> Node:
        return self._root

    def searchPoint(self, point: Vector, node: Node = None) -> Optional[Node]:
        if node is None:
            node = self._root

        if node.isLeaf:
            return node

        isInside = None
        for child in node.children:
            if child.bbox.contains(point):
                isInside = self.searchPoint(point, child)

        if isInside is None:
            if node.bbox.contains(point):
                return node

            else:
                return None
        return isInside

    def getNodeCount(self, node=None) -> int:
        if node is None:
            node = self._root
        counter = 1
        for childNode in node.children:
            counter += self.getNodeCount(childNode)

        return counter

    def getLeafCount(self, node=None) -> int:
        if node is None:
            node = self._root
        counter = 0
        if node.isLeaf:
            counter += 1
        else:
            for childNode in node.children:
                counter += self.getLeafCount(childNode)
        return counter

    def getLeafNodes(self, node=None, nodesList=None) -> List[Node]:
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

    def getMaxDepth(self) -> int:
        leaves = self.getLeafNodes()
        maxDepth = 0
        for leaf in leaves:
            if leaf.depth > maxDepth:
                maxDepth = leaf.depth
        return maxDepth

    def getAverageDepth(self) -> float:
        leaves = self.getLeafNodes()
        avgDepth = 0
        for leaf in leaves:
            avgDepth += leaf.depth
        avgDepth = avgDepth / len(leaves)
        return avgDepth

    def getAverageLeafSize(self) -> float:
        leaves = self.getLeafNodes()
        avgSize = 0
        for leaf in leaves:
            avgSize += len(leaf.polygons)
        avgSize = avgSize / len(leaves)
        return avgSize

    def getLeafPolygons(self, node=None) -> List[Polygon]:
        if node is None:
            node = self._root
        polygons = []
        if node.isLeaf:
            polygons = node.polygons
        else:
            for childNode in node.children:
                polygons.extend(self.getLeafPolygons(childNode))
        return polygons
