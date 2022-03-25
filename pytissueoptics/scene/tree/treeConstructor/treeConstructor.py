from pytissueoptics.scene.tree import Node
from pytissueoptics.scene.tree.treeConstructor import PolygonCounter, AxisSelector, NodeSplitter, SplitNodeResult


class TreeConstructor:
    def __init__(self):
        self._polyCounter: PolygonCounter = None
        self._axisSelector: AxisSelector = None
        self._nodeSplitter: NodeSplitter = None

    def setContext(self, axisSelector: AxisSelector, polyCounter: PolygonCounter, nodeSplitter: NodeSplitter):
        self._axisSelector = axisSelector
        self._polyCounter = polyCounter
        self._nodeSplitter = nodeSplitter
        self._nodeSplitter.setContext(self._polyCounter)

    def _splitNode(self, node: Node) -> SplitNodeResult:
        nodeDepth = node.depth
        nodeBbox = node.bbox
        nodePolygons = node.polygons
        splitAxis = self._axisSelector.select(nodeDepth, nodeBbox, nodePolygons)
        splitNodeResult = self._nodeSplitter.split(splitAxis, nodeBbox, nodePolygons)
        return splitNodeResult

    def growTree(self, node: Node, maxDepth: int, minLeafSize: int):
        if node.depth >= maxDepth or len(node.polygons) <= minLeafSize:
            return

        splitNodeResult = self._splitNode(node)
        if splitNodeResult.stopCondition:
            return

        for i, polygonGroup in enumerate(splitNodeResult.polygonGroups):
            if len(polygonGroup) <= 0:
                continue
            childNode = Node(parent=node, polygons=polygonGroup, bbox=splitNodeResult.groupsBbox[i], depth=node.depth + 1)
            node.children.append(childNode)
            self.growTree(childNode, maxDepth, minLeafSize)
