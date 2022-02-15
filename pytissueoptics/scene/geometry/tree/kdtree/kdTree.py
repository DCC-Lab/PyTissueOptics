from typing import List
from pytissueoptics.scene.geometry import BoundingBox
from pytissueoptics.scene.geometry.tree.kdtree.splitter import Splitter
from pytissueoptics.scene.geometry.tree.kdtree import KDNode
from pytissueoptics.scene.solids import Cuboid
from pytissueoptics.scene.scene import Scene


class KDTree:
    def __init__(self, scene: Scene, splitStrategy: Splitter, maxDepth: int):
        self._scene = scene
        self._maxDepth = maxDepth
        self._splitStrategy = splitStrategy
        self._root = KDNode(scene=scene, splitStrategy=splitStrategy, maxDepth=maxDepth)

    def search(self):
        raise NotImplementedError

    def _getBoundingBoxes(self) -> List[BoundingBox]:
        boundingBoxes = self._root.getBoundingBoxes(self._root, bboxList=[])
        return boundingBoxes

    def getBoundingBoxesAsCuboids(self):
        cuboids = []
        for bbox in self._getBoundingBoxes():
            a = bbox.xMax-bbox.xMin
            b = bbox.yMax-bbox.yMin
            c = bbox.zMax-bbox.zMin
            cuboids.append(Cuboid(a=a, b=b, c=c, position=bbox.center))
