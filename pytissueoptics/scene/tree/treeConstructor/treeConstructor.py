from pytissueoptics.scene.geometry import BoundingBox
from pytissueoptics.scene.tree import Node
from pytissueoptics.scene.tree.treeConstructor import SplitNodeResult


class TreeConstructor:
    EPSILON = 1e-6

    def _splitNode(self, node: Node) -> SplitNodeResult:
        raise NotImplementedError()

    def constructTree(self, node: Node, maxDepth: int, minLeafSize: int):
        if node.depth >= maxDepth or len(node.polygons) <= minLeafSize:
            return

        splitNodeResult = self._splitNode(node)
        if splitNodeResult.stopCondition:
            return

        for i, polygonGroup in enumerate(splitNodeResult.polygonGroups):
            if len(polygonGroup) <= 0:
                continue
            childNode = Node(parent=node, polygons=polygonGroup, bbox=splitNodeResult.groupsBbox[i],
                             depth=node.depth + 1)
            node.children.append(childNode)
            self.constructTree(childNode, maxDepth, minLeafSize)
