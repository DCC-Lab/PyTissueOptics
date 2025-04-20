import sys

from pytissueoptics.scene.tree import Node
from pytissueoptics.scene.tree.treeConstructor import SplitNodeResult
from pytissueoptics.scene.tree.treeConstructor.binary.noSplitOneAxisConstructor import NoSplitOneAxisConstructor


class NoSplitThreeAxesConstructor(NoSplitOneAxisConstructor):
    def _splitNode(self, node: Node) -> SplitNodeResult:
        self.currentNode = node
        splitBbox = self.currentNode.bbox.copy()
        minSAH = sys.float_info.max
        for axis in ["x", "y", "z"]:
            thisSAH = self._searchMinSAHOnAxis(splitBbox, axis, minSAH)
            if thisSAH < minSAH:
                minSAH = thisSAH
        self.result.leftPolygons.extend(self.result.splitPolygons)
        self.result.rightPolygons.extend(self.result.splitPolygons)
        self._trimChildrenBbox()
        stopCondition = self._checkStopCondition()
        newNodeResult = SplitNodeResult(
            stopCondition,
            [self.result.leftBbox, self.result.rightBbox],
            [self.result.leftPolygons, self.result.rightPolygons],
        )
        return newNodeResult
