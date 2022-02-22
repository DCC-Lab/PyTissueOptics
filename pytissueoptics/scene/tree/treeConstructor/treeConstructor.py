from pytissueoptics.scene.tree import Node
from pytissueoptics.scene.tree.treeConstructor import PolyCounter, AxisSelector, NodeSplitter, SplitNodeResult


class TreeConstructor:
    def __init__(self):
        self._polyCounter: PolyCounter = None
        self._axisSelector: AxisSelector = None
        self._nodeSplitter: NodeSplitter = None

    def setContext(self, axisSelector: AxisSelector, polyCounter: PolyCounter, nodeSplitter: NodeSplitter):
        self._axisSelector = axisSelector
        self._polyCounter = polyCounter
        self._nodeSplitter = nodeSplitter

    def _splitNode(self, node: Node) -> SplitNodeResult:
        nodeDepth = node.depth
        nodeBbox = node.bbox
        nodePolygons = node.polygons
        splitAxis = self._axisSelector.run(nodeDepth, nodeBbox, nodePolygons)
        splitNodeResult = self._nodeSplitter.run(splitAxis, nodeBbox, nodePolygons)
        return splitNodeResult

    def growTree(self, node):
        if node.depth < node.maxDepth and len(node.polygons) > node.maxLeafSize:
            splitNodeResult = self._splitNode(node)
            if not splitNodeResult.stopCondition:
                for i, polygonGroup in enumerate(splitNodeResult.polygonGroups):
                    if len(polygonGroup) > 0:
                        childNode = Node(parent=node, polygons=polygonGroup,
                                         bbox=splitNodeResult.bboxes[i], depth=node.depth + 1,
                                         maxDepth=node.maxDepth, maxLeafSize=node.maxLeafSize)
                        node.children.append(childNode)
                        self.growTree(childNode)
