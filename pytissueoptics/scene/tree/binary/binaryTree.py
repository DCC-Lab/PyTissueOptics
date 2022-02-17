from typing import List
from pytissueoptics.scene.geometry import BoundingBox, Vector
from pytissueoptics.scene.tree.binary import BinaryTreeStrategy
from pytissueoptics.scene.tree.binary import BinaryNode
from pytissueoptics.scene.solids import Cuboid
from pytissueoptics.scene.scene import Scene


class BinaryTree:
    def __init__(self, scene: Scene, treeStrategy: BinaryTreeStrategy, maxDepth: int):
        self._scene = scene
        self._maxDepth = maxDepth
        self._splitStrategy = treeStrategy
        self._root = BinaryNode(scene=scene, treeStrategy=treeStrategy, maxDepth=maxDepth)

    def searchPoint(self, point: Vector) -> BoundingBox:
        return self._root.searchPoint(point)

    def _getBoundingBoxes(self) -> List[BoundingBox]:
        boundingBoxes = self._root.getLeafBoundingBoxes(self._root, bboxList=[])
        return boundingBoxes

    def getLeafBoundingBoxesAsCuboids(self) -> List[Cuboid]:
        cuboids = []
        for bbox in self._getBoundingBoxes():
            a = bbox.xMax-bbox.xMin
            b = bbox.yMax-bbox.yMin
            c = bbox.zMax-bbox.zMin
            cuboids.append(Cuboid(a=a, b=b, c=c, position=bbox.center))
        return cuboids
